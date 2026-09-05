"""Sequence, control-flow, and random operator tests, without model-specific patterns."""
import numpy as np
from onnx import TensorProto as T, helper


def register(case, model, tin, const, f32):
    @case
    def if_lexical_capture():
        branches = {
            'then_branch': helper.make_graph([helper.make_node('Add', ['x', 'delta'], ['a'])], 'then', [], [tin('a', [2, 3])]),
            'else_branch': helper.make_graph([helper.make_node('Mul', ['x', 'delta'], ['b'])], 'else', [], [tin('b', [2, 3])]),
        }
        node = helper.make_node('If', ['cond'], ['y'], **branches)
        m = model([node], [tin('x', [2, 3]), tin('cond', [], T.BOOL)], [tin('y', [2, 3])], [const('delta', np.float32(2))])
        x = f32(2, 3)
        return m, [{'x': x, 'cond': np.array(v, dtype=np.bool_)} for v in [True, False, True]], 0, 0

    @case
    def sequence_split_insert_concat():
        nodes = [
            helper.make_node('SplitToSequence', ['x', 'split'], ['seq'], axis=1),
            helper.make_node('Identity', ['seq'], ['copy']),
            helper.make_node('SequenceLength', ['copy'], ['length']),
            helper.make_node('SequenceAt', ['seq', 'idx'], ['last']),
            helper.make_node('SequenceEmpty', [], ['empty'], dtype=T.FLOAT),
            helper.make_node('SequenceInsert', ['empty', 'last'], ['one']),
            helper.make_node('SequenceInsert', ['one', 'last', 'zero'], ['two']),
            helper.make_node('ConcatFromSequence', ['seq'], ['restored'], axis=1),
            helper.make_node('ConcatFromSequence', ['two'], ['stack'], axis=0, new_axis=1),
        ]
        inits = [const('split', np.int64(2)), const('idx', np.int64(-1)), const('zero', np.int64(0))]
        return model(nodes, [tin('x', [2, 5, 3])], [tin('restored', [2, 5, 3]), tin('stack', [2, 2, 1, 3]), tin('length', [], T.INT64)], inits), {'x': f32(2, 5, 3)}, 0, 0

    @case
    def sequence_split_squeeze():
        nodes = [helper.make_node('SplitToSequence', ['x'], ['s'], axis=-1, keepdims=0),
                 helper.make_node('ConcatFromSequence', ['s'], ['y'], axis=-1, new_axis=1)]
        return model(nodes, [tin('x', [2, 3, 4])], [tin('y', [2, 3, 4])]), {'x': f32(2, 3, 4)}, 0, 0

    @case
    def loop_carried_and_scan():
        body = helper.make_graph([
            helper.make_node('Add', ['state', 'delta'], ['next']),
            helper.make_node('Identity', ['cond_in'], ['cond_out']),
            helper.make_node('Identity', ['next'], ['scan']),
        ], 'body', [tin('iteration', [], T.INT64), tin('cond_in', [], T.BOOL), tin('state', [3])],
            [tin('cond_out', [], T.BOOL), tin('next', [3]), tin('scan', [3])])
        node = helper.make_node('Loop', ['count', 'cond', 'x'], ['y', 'scan_out'], body=body)
        m = model([node], [tin('count', [], T.INT64), tin('cond', [], T.BOOL), tin('x', [3])], [tin('y', [3]), tin('scan_out', ['steps', 3])], [const('delta', np.float32(0.5))])
        return m, [{'count': np.array(count, dtype=np.int64), 'cond': np.array(cond, dtype=np.bool_), 'x': np.array([1, 2, 3], np.float32)} for count, cond in [(4, True), (0, True), (4, False)]], 0, 0

    @case
    def loop_sequence_carried():
        seq_info = lambda name: helper.make_tensor_sequence_value_info(name, T.FLOAT, [2])
        body = helper.make_graph([
            helper.make_node('SequenceInsert', ['carried', 'x'], ['next']),
            helper.make_node('Identity', ['cond_in'], ['cond_out']),
        ], 'body', [tin('iteration', [], T.INT64), tin('cond_in', [], T.BOOL), seq_info('carried')],
            [tin('cond_out', [], T.BOOL), seq_info('next')])
        nodes = [helper.make_node('SequenceEmpty', [], ['empty'], dtype=T.FLOAT),
                 helper.make_node('Loop', ['count', 'cond', 'empty'], ['sequence'], body=body),
                 helper.make_node('ConcatFromSequence', ['sequence'], ['y'], axis=0, new_axis=1)]
        inits = [const('count', np.int64(3)), const('cond', np.bool_(True))]
        return model(nodes, [tin('x', [2])], [tin('y', [3, 2])], inits), {'x': f32(2)}, 0, 0

    @case
    def random_normal_zero_scale():
        node = helper.make_node('RandomNormalLike', ['x'], ['y'], seed=42., mean=0.25, scale=0.)
        return model([node], [tin('x', [3, 7])], [tin('y', [3, 7])]), {'x': f32(3, 7)}, 0, 0

    @case
    def random_uniform_bounds():
        nodes = [helper.make_node('RandomUniformLike', ['x'], ['noise'], seed=42., low=-2., high=3.),
                 helper.make_node('GreaterOrEqual', ['noise', 'low'], ['above']),
                 helper.make_node('Less', ['noise', 'high'], ['below']),
                 helper.make_node('And', ['above', 'below'], ['y'])]
        inits = [const('low', np.float32(-2)), const('high', np.float32(3))]
        return model(nodes, [tin('x', [32, 32])], [tin('y', [32, 32], T.BOOL)], inits), {'x': f32(32, 32)}, 0, 0
