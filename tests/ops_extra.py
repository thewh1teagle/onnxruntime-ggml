"""Independent standard ONNX operator cases, registered by test_ops.py."""
import numpy as np
from onnx import TensorProto as T, helper


def register(case, model, tin, const, f32):
    def single(name, op, feeds, output_shapes, inits=(), attrs=None, output_types=None, atol=1e-5, rtol=1e-5):
        def run():
            types = {np.dtype('float32'): T.FLOAT, np.dtype('float64'): T.DOUBLE,
                     np.dtype('int64'): T.INT64, np.dtype('int32'): T.INT32, np.dtype('bool'): T.BOOL}
            outputs = [f'y{i}' for i in range(len(output_shapes))]
            node = helper.make_node(op, list(feeds) + [i.name for i in inits], outputs, **(attrs or {}))
            ins = [tin(k, v.shape, types[v.dtype]) for k, v in feeds.items()]
            outs = [tin(k, shape, (output_types or [T.FLOAT] * len(outputs))[i]) for i, (k, shape) in enumerate(zip(outputs, output_shapes))]
            return model([node], ins, outs, inits), feeds, atol, rtol
        run.__name__ = name
        case(run)

    single('leaky_relu_alpha', 'LeakyRelu', {'x': np.array([-4, -0., 0., 3], np.float32)}, [[4]], attrs={'alpha': 0.2})
    single('round_ties_even', 'Round', {'x': np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], np.float32)}, [[8]], atol=0, rtol=0)
    single('atan_range', 'Atan', {'x': np.array([-1e5, -4, -1, 0, 1, 4, 1e5], np.float32)}, [[7]])
    single('isnan_special_values', 'IsNaN', {'x': np.array([0., np.nan, np.inf, -np.inf, -1.], np.float32)}, [[5]], output_types=[T.BOOL], atol=0, rtol=0)
    for shape in [(2, 3, 7), (2, 3, 4, 5)]:
        single(f'instance_norm_{len(shape)}d', 'InstanceNormalization', {'x': f32(*shape)}, [list(shape)],
               [const('scale', f32(3)), const('bias', f32(3))], {'epsilon': 1e-4}, atol=2e-5, rtol=1e-4)
    for axis in [-1, 0, 2, 3]:
        ax = axis if axis >= 0 else 3 + axis
        single(f'flatten_axis_{axis}', 'Flatten', {'x': f32(2, 3, 4)}, [[int(np.prod((2, 3, 4)[:ax])), int(np.prod((2, 3, 4)[ax:]))]], attrs={'axis': axis}, atol=0, rtol=0)
    single('gather_axis1_multi_index', 'Gather', {'x': f32(2, 4, 3)}, [[2, 2, 2, 3]], [const('ids', np.array([[3, 1], [-1, 0]], np.int64))], {'axis': 1}, atol=0, rtol=0)
    single('gather_elements_negative', 'GatherElements', {'x': f32(2, 4, 3)}, [[2, 2, 3]], [const('ids', np.array([[[0, 1, -1], [2, 0, 1]], [[-1, 0, 2], [1, 1, -2]]], np.int64))], {'axis': 1}, atol=0, rtol=0)
    single('scatter_elements_axis1', 'ScatterElements', {'x': f32(2, 4)}, [[2, 4]],
           [const('ids', np.array([[0, -1], [1, 2]], np.int64)), const('updates', f32(2, 2))], {'axis': 1}, atol=0, rtol=0)
    single('scatter_nd_slices', 'ScatterND', {'x': f32(2, 3, 4)}, [[2, 3, 4]],
           [const('ids', np.array([[0, -1], [1, 0]], np.int64)), const('updates', f32(2, 4))], atol=0, rtol=0)
    for largest in [0, 1]:
        single(f'topk_axis0_{largest}', 'TopK', {'x': np.array([[2, 1], [2, 3], [0, 3], [5, -1]], np.float32)}, [[2, 2], [2, 2]],
               [const('k', np.array([2], np.int64))], {'axis': 0, 'largest': largest}, output_types=[T.FLOAT, T.INT64], atol=0, rtol=0)
    for axis in [0, 1, -1]:
        for reverse in [0, 1]:
            for exclusive in [0, 1]:
                single(f'cumsum_{axis}_{reverse}_{exclusive}', 'CumSum', {'x': f32(2, 3, 5)}, [[2, 3, 5]],
                       [const('axis', np.int64(axis))], {'reverse': reverse, 'exclusive': exclusive}, atol=2e-6)
    for mode in ['constant', 'edge', 'reflect']:
        single(f'pad_{mode}', 'Pad', {'x': f32(2, 3, 4)}, [[3, 5, 7]],
               [const('pads', np.array([1, 1, 2, 0, 1, 1], np.int64))], {'mode': mode}, atol=0, rtol=0)
    single('pad_crop_fill', 'Pad', {'x': f32(2, 3, 4)}, [[2, 2, 6]],
           [const('pads', np.array([0, -1, 1, 0, 0, 1], np.int64)), const('fill', np.float32(2.5))], {'mode': 'constant'}, atol=0, rtol=0)

    def resize_case(mode, coordinate, nearest='floor', sizes=False):
        def run():
            x = f32(2, 3, 7)
            attrs = {'mode': mode, 'coordinate_transformation_mode': coordinate, 'nearest_mode': nearest}
            inputs = ['x', '', '', 'sizes'] if sizes else ['x', '', 'scales']
            init = const('sizes', np.array([2, 5, 4], np.int64)) if sizes else const('scales', np.array([1., 1.5, 2.3], np.float32))
            shape = [2, 5, 4] if sizes else [2, 4, 16]
            return model([helper.make_node('Resize', inputs, ['y'], **attrs)], [tin('x', x.shape)], [tin('y', shape)], [init]), {'x': x}, 2e-5, 1e-4
        run.__name__ = f'resize_{mode}_{coordinate}_{nearest}_{sizes}'
        case(run)
    for coordinate in ['asymmetric', 'half_pixel', 'align_corners', 'pytorch_half_pixel']:
        resize_case('linear', coordinate)
        resize_case('linear', coordinate, sizes=True)
    for nearest in ['floor', 'ceil', 'round_prefer_floor', 'round_prefer_ceil']:
        resize_case('nearest', 'half_pixel', nearest)
        resize_case('nearest', 'asymmetric', nearest, sizes=True)

    single('cumsum_int64_exact', 'CumSum', {'x': np.array([2**54, 1, -2**54, 7], np.int64)}, [[4]],
           [const('axis', np.int64(0))], output_types=[T.INT64], atol=0, rtol=0)
    single('topk_int64_exact', 'TopK', {'x': np.array([2**54, 2**54 + 1, 2**54 - 1], np.int64)}, [[2], [2]],
           [const('k', np.array([2], np.int64))], output_types=[T.INT64, T.INT64], atol=0, rtol=0)

    def lstm_case(direction, extras=False):
        def run():
            s, batch, inputs, hidden = 5, 2, 3, 4
            dirs = 2 if direction == 'bidirectional' else 1
            x = f32(s, batch, inputs) * 0.2
            names = ['x', 'w', 'r', 'b', 'lens', 'h', 'c']
            inits = [const('w', f32(dirs, 4 * hidden, inputs) * 0.2), const('r', f32(dirs, 4 * hidden, hidden) * 0.2),
                     const('b', f32(dirs, 8 * hidden) * 0.2), const('lens', np.array([5, 3], np.int32)),
                     const('h', f32(dirs, batch, hidden) * 0.1), const('c', f32(dirs, batch, hidden) * 0.1)]
            attrs = {'direction': direction, 'hidden_size': hidden}
            if extras:
                names.append('p'); inits.append(const('p', f32(dirs, 3 * hidden) * 0.2))
                attrs.update(input_forget=1, clip=0.1)
            node = helper.make_node('LSTM', names, ['y', 'yh', 'yc'], **attrs)
            outs = [tin('y', [s, dirs, batch, hidden]), tin('yh', [dirs, batch, hidden]), tin('yc', [dirs, batch, hidden])]
            return model([node], [tin('x', x.shape)], outs, inits), {'x': x}, 2e-5, 1e-4
        run.__name__ = f'lstm_{direction}_{extras}'
        case(run)
    for direction in ['forward', 'reverse', 'bidirectional']:
        lstm_case(direction)
        lstm_case(direction, extras=True)

    for groups, channels in [(1, 3), (3, 3)]:
        single(f'conv_transpose_output_padding_group{groups}', 'ConvTranspose', {'x': f32(1, channels, 5)}, [[1, 3, 10]],
               [const('w', f32(channels, 3 // groups, 3)), const('b', f32(3))],
               {'kernel_shape': [3], 'pads': [1, 1], 'strides': [2], 'output_padding': [1], 'group': groups}, atol=2e-5, rtol=1e-4)

    for batch in [1, 2]:
        single(f'conv_transpose_no_bias_batch{batch}', 'ConvTranspose', {'x': f32(batch, 3, 5)}, [[batch, 4, 12]],
               [const('w', f32(3, 4, 4))], {'strides': [2]}, atol=2e-5, rtol=1e-4)
    single('conv_batch2', 'Conv', {'x': f32(2, 3, 9)}, [[2, 4, 9]],
           [const('w', f32(4, 3, 3)), const('b', f32(4))], {'pads': [1, 1]}, atol=2e-5, rtol=1e-4)
    for exponent in [-3, -1, 0, 1, 2, 3, 8]:
        single(f'pow_scalar_{exponent}', 'Pow', {'x': np.array([[-2., -0.5, 0.5, 2.]], np.float32)}, [[1, 4]],
               [const('exponent', np.float32(exponent))], atol=1e-6, rtol=1e-5)

    for inner in [70, 96]:
        def precise_matmul(inner=inner):
            x = f32(35, inner)
            w = f32(inner, 67)
            nodes = [helper.make_node('MatMul', ['x', 'w'], ['y'])]
            return model(nodes, [tin('x', x.shape)], [tin('y', [35, 67])], [const('w', w)]), {'x': x}, 3e-5, 2e-5, {'weights': 'f32'}
        precise_matmul.__name__ = f'matmul_f32_precision_{inner}'
        case(precise_matmul)

    single('conv_transpose_wide_depthwise', 'ConvTranspose', {'x': f32(1, 1031, 7)}, [[1, 1031, 14]],
           [const('w', f32(1031, 1, 3)), const('b', f32(1031))],
           {'strides': [2], 'pads': [1, 1], 'output_padding': [1], 'group': 1031}, atol=2e-5, rtol=1e-4)

    single('conv_strided_dilated_asymmetric', 'Conv', {'x': f32(2, 7, 23)}, [[2, 5, 9]],
           [const('w', f32(5, 7, 5)), const('b', f32(5))],
           {'strides': [2], 'dilations': [2], 'pads': [1, 2]}, atol=3e-5, rtol=1e-4)
    single('conv_large_kernel', 'Conv', {'x': f32(1, 2, 1041)}, [[1, 3, 9]],
           [const('w', f32(3, 2, 1033))], atol=2e-4, rtol=1e-4)
