use super::{Error, buffer_bytes, validate};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};
use objc::{msg_send, sel, sel_impl};

#[derive(Debug, Clone, Copy)]
pub struct GpuTiming {
    /// Encoder creation, dispatch encoding, submission and synchronous wait.
    /// Excludes input validation/copies and output validation/copy.
    pub submission_wait_ns: u128,
    /// Command-buffer GPU timestamps, excluding CPU copies and waits. None if
    /// the driver provides no positive finite interval; never invent a zero.
    pub seconds: Option<f64>,
}

/// One caller, one synchronous command at a time. Metal objects retain storage;
/// &mut self excludes overlapping CPU mappings and submissions in this wrapper.
/// Rust slice memory is copied, never lent to an asynchronous GPU command.
pub struct ResidualAdd {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    x: Buffer,
    residual: Buffer,
    y: Buffer,
    capacity: usize,
    group_width: usize,
    usable: bool,
}

impl ResidualAdd {
    pub fn new(capacity: usize) -> Result<Self, Error> {
        let bytes = buffer_bytes(capacity)?;
        objc::rc::autoreleasepool(|| {
            let device = Device::system_default().ok_or(Error::Unavailable("no Metal device"))?;
            if bytes as u64 > device.max_buffer_length() {
                return Err(Error::TooLarge);
            }
            let options = metal::CompileOptions::new();
            options.set_fast_math_enabled(false);
            let library = device
                .new_library_with_source(include_str!("residual.metal"), &options)
                .map_err(Error::Gpu)?;
            let function = library
                .get_function("residual_add", None)
                .map_err(Error::Gpu)?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(Error::Gpu)?;
            let group_width = (pipeline.thread_execution_width() as usize)
                .min(pipeline.max_total_threads_per_threadgroup() as usize);
            if group_width == 0 {
                return Err(Error::Unavailable("zero threadgroup width"));
            }
            let x = shared_buffer(&device, bytes)?;
            let residual = shared_buffer(&device, bytes)?;
            let y = shared_buffer(&device, bytes)?;
            for buffer in [&x, &residual, &y] {
                if buffer.contents().is_null() || buffer.length() < bytes as u64 {
                    return Err(Error::Unavailable("shared buffer allocation failed"));
                }
                if !(buffer.contents() as usize).is_multiple_of(align_of::<f32>()) {
                    return Err(Error::Unavailable("unaligned shared buffer"));
                }
            }
            // SAFETY: documented no-argument selector returns an owned (+1)
            // MTLCommandQueue. Check nil before constructing a Rust owner.
            let queue = unsafe {
                let pointer: *mut metal::MTLCommandQueue = msg_send![&*device, newCommandQueue];
                if pointer.is_null() {
                    return Err(Error::Unavailable("command queue allocation failed"));
                }
                CommandQueue::from_ptr(pointer)
            };
            Ok(Self {
                device,
                queue,
                pipeline,
                x,
                residual,
                y,
                capacity,
                group_width,
                usable: true,
            })
        })
    }
    pub fn device_name(&self) -> &str {
        self.device.name()
    }
    pub fn has_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }
    pub fn group_width(&self) -> usize {
        self.group_width
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Compute finite FP32 residual addition into caller-owned output. Every
    /// error preserves y. Finite arithmetic overflow rejects before copying y.
    /// Metal may flush subnormal arithmetic; bitwise CPU identity is not promised.
    pub fn run(&mut self, x: &[f32], residual: &[f32], y: &mut [f32]) -> Result<GpuTiming, Error> {
        validate(x, residual, y.len(), self.capacity)?;
        if !self.usable {
            return Err(Error::Unavailable("command failure invalidated executor"));
        }
        objc::rc::autoreleasepool(|| {
            // SAFETY: buffers own capacity*4 allocated shared bytes, are aligned
            // for f32, and validate proved n<=capacity. Live Rust inputs cannot
            // alias these private buffers. &mut self and prior synchronous wait
            // exclude GPU use while copying; both CPU ranges are initialized.
            unsafe {
                std::ptr::copy_nonoverlapping(x.as_ptr(), self.x.contents().cast::<f32>(), x.len());
                std::ptr::copy_nonoverlapping(
                    residual.as_ptr(),
                    self.residual.contents().cast::<f32>(),
                    residual.len(),
                );
            }
            let start = std::time::Instant::now();
            // SAFETY: documented selectors return autoreleased objects, kept
            // alive by this pool and retained parent until wait completes. Nil
            // is checked before creating borrowed references. No reference escapes.
            let (command, encoder) = unsafe {
                let pointer: *mut metal::MTLCommandBuffer = msg_send![&*self.queue, commandBuffer];
                if pointer.is_null() {
                    return Err(Error::Unavailable("command buffer allocation failed"));
                }
                let command = metal::CommandBufferRef::from_ptr(pointer);
                let pointer: *mut metal::MTLComputeCommandEncoder =
                    msg_send![command, computeCommandEncoder];
                if pointer.is_null() {
                    return Err(Error::Unavailable("compute encoder allocation failed"));
                }
                (command, metal::ComputeCommandEncoderRef::from_ptr(pointer))
            };
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&self.x), 0);
            encoder.set_buffer(1, Some(&self.residual), 0);
            encoder.set_buffer(2, Some(&self.y), 0);
            let n = x.len() as u32;
            encoder.set_bytes(3, size_of::<u32>() as u64, (&n as *const u32).cast());
            let groups = x.len().div_ceil(self.group_width);
            encoder.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(self.group_width as u64, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            let submission_wait_ns = start.elapsed().as_nanos();
            if command.status() != MTLCommandBufferStatus::Completed {
                self.usable = false;
                return Err(Error::Gpu(format!("command status {:?}", command.status())));
            }
            // SAFETY: retained shared storage is aligned and has n initialized
            // outputs: successful command writes each index<n exactly once.
            // Completion ends GPU writes; &mut self prevents another submission.
            // The resulting slice never escapes this synchronous method.
            let output =
                unsafe { std::slice::from_raw_parts(self.y.contents().cast::<f32>(), y.len()) };
            if let Some(index) = output.iter().position(|v| !v.is_finite()) {
                return Err(Error::NonFinite {
                    buffer: "GPU output",
                    index,
                });
            }
            y.copy_from_slice(output);
            // SAFETY: documented MTLCommandBuffer selectors return CFTimeInterval
            // (double/f64). The command is live and successfully completed.
            let seconds = unsafe {
                let end: f64 = msg_send![command, GPUEndTime];
                let start: f64 = msg_send![command, GPUStartTime];
                end - start
            };
            Ok(GpuTiming {
                submission_wait_ns,
                seconds: (seconds.is_finite() && seconds > 0.0).then_some(seconds),
            })
        })
    }
}

fn shared_buffer(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    // SAFETY: documented selector takes NSUInteger/MTLResourceOptions and
    // returns an owned (+1) MTLBuffer. Bytes are checked against device limits.
    // Raw pointer return handles allocation nil before any Rust owner exists.
    unsafe {
        let pointer: *mut metal::MTLBuffer = msg_send![&**device, newBufferWithLength:bytes as u64 options:MTLResourceOptions::StorageModeShared];
        if pointer.is_null() {
            return Err(Error::Unavailable("shared buffer allocation failed"));
        }
        Ok(Buffer::from_ptr(pointer))
    }
}
