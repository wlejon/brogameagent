#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

namespace brogameagent::nn::gpu {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::dispatch1d_sync;

namespace {

void launch_unary(NSString* name,
                  const GpuTensor& in, GpuTensor& out) {
    if (out.rows != in.rows || out.cols != in.cols) out.resize(in.rows, in.cols);
    const uint32_t n = static_cast<uint32_t>(in.size());
    if (n == 0) return;
    id<MTLBuffer> bin  = buffer_for(in);
    id<MTLBuffer> bout = buffer_for(out);
    const NSUInteger off_in  = buffer_offset_for(in);
    const NSUInteger off_out = buffer_offset_for(out);
    dispatch1d_sync(name, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bin offset:off_in atIndex:0];
        [enc setBuffer:bout offset:off_out atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void launch_binary_back(NSString* name,
                        const GpuTensor& a, const GpuTensor& dY,
                        GpuTensor& dX, int rows, int cols) {
    if (dX.rows != rows || dX.cols != cols) dX.resize(rows, cols);
    const uint32_t n = static_cast<uint32_t>(rows * cols);
    if (n == 0) return;
    id<MTLBuffer> ba  = buffer_for(a);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger off_a  = buffer_offset_for(a);
    const NSUInteger off_dy = buffer_offset_for(dY);
    const NSUInteger off_dx = buffer_offset_for(dX);
    dispatch1d_sync(name, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba  offset:off_a  atIndex:0];
        [enc setBuffer:bdy offset:off_dy atIndex:1];
        [enc setBuffer:bdx offset:off_dx atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    });
}

} // namespace

void relu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    launch_unary(@"k_relu_forward", x, y);
}
void relu_backward_gpu(const GpuTensor& x, const GpuTensor& dY, GpuTensor& dX) {
    launch_binary_back(@"k_relu_backward", x, dY, dX, x.rows, x.cols);
}
void tanh_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    launch_unary(@"k_tanh_forward", x, y);
}
void tanh_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX) {
    launch_binary_back(@"k_tanh_backward", y, dY, dX, y.rows, y.cols);
}
void sigmoid_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    launch_unary(@"k_sigmoid_forward", x, y);
}
void sigmoid_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX) {
    launch_binary_back(@"k_sigmoid_backward", y, dY, dX, y.rows, y.cols);
}

void add_inplace_gpu(GpuTensor& y, const GpuTensor& x) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger off_y = buffer_offset_for(y);
    const NSUInteger off_x = buffer_offset_for(x);
    dispatch1d_sync(@"k_add_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBuffer:bx offset:off_x atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void add_scalar_inplace_gpu(GpuTensor& y, float s) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    dispatch1d_sync(@"k_add_scalar_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void scale_inplace_gpu(GpuTensor& y, float s) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    dispatch1d_sync(@"k_scale_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

} // namespace brogameagent::nn::gpu
