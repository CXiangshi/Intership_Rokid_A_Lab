       �K"	  �����Abrain.Event:2L��Md     <�}j	Ŀ����A"��	
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:@���������*
shape:@���������
v
input/Placeholder_1Placeholder*
shape:@���������*
dtype0*'
_output_shapes
:@���������
i
$learning_rate/Variable/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
learning_rate/Variable
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
learning_rate/Variable/AssignAssignlearning_rate/Variable$learning_rate/Variable/initial_value*
use_locking(*
T0*)
_class
loc:@learning_rate/Variable*
validate_shape(*
_output_shapes
: 
�
learning_rate/Variable/readIdentitylearning_rate/Variable*
T0*)
_class
loc:@learning_rate/Variable*
_output_shapes
: 
d
learning_rate_1/tagsConst* 
valueB Blearning_rate_1*
dtype0*
_output_shapes
: 
t
learning_rate_1ScalarSummarylearning_rate_1/tagslearning_rate/Variable/read*
T0*
_output_shapes
: 
�
;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
valueB:@*
dtype0
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
<rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concatConcatV2;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_1Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zerosFill<rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concatArnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/Const*
_output_shapes
:	@�*
T0*

index_type0
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:@*
dtype0*
_output_shapes
:
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:�
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:@*
dtype0*
_output_shapes
:
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
valueB:�*
dtype0
�
Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1ConcatV2=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_4=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_5Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1Fill>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes
:	@�*
T0*

index_type0
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:@*
dtype0*
_output_shapes
:
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:�*
dtype0*
_output_shapes
:
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ConstConst*
valueB:@*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatConcatV2=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_1Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zerosFill>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatCrnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/Const*
T0*

index_type0*
_output_shapes
:	@�
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_2Const*
valueB:@*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_3Const*
_output_shapes
:*
valueB:�*
dtype0
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_4Const*
valueB:@*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_5Const*
valueB:�*
dtype0*
_output_shapes
:
�
Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
@rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1ConcatV2?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_4?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_5Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1Fill@rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	@�
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_6Const*
valueB:@*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_7Const*
dtype0*
_output_shapes
:*
valueB:�
�
*embedding/Initializer/random_uniform/shapeConst*
_class
loc:@embedding*
valueB"�  �   *
dtype0*
_output_shapes
:
�
(embedding/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *Y��
�
(embedding/Initializer/random_uniform/maxConst*
_class
loc:@embedding*
valueB
 *Y�<*
dtype0*
_output_shapes
: 
�
2embedding/Initializer/random_uniform/RandomUniformRandomUniform*embedding/Initializer/random_uniform/shape*
T0*
_class
loc:@embedding*
seed2 *
dtype0* 
_output_shapes
:
�/�*

seed 
�
(embedding/Initializer/random_uniform/subSub(embedding/Initializer/random_uniform/max(embedding/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@embedding
�
(embedding/Initializer/random_uniform/mulMul2embedding/Initializer/random_uniform/RandomUniform(embedding/Initializer/random_uniform/sub*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
$embedding/Initializer/random_uniformAdd(embedding/Initializer/random_uniform/mul(embedding/Initializer/random_uniform/min*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
	embedding
VariableV2*
dtype0* 
_output_shapes
:
�/�*
shared_name *
_class
loc:@embedding*
	container *
shape:
�/�
�
embedding/AssignAssign	embedding$embedding/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�
n
embedding/readIdentity	embedding*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
x
lm/embedding_lookup/axisConst*
_class
loc:@embedding*
value	B : *
dtype0*
_output_shapes
: 
�
lm/embedding_lookupGatherV2embedding/readinput/Placeholderlm/embedding_lookup/axis*,
_output_shapes
:@����������*
Taxis0*
Tindices0*
Tparams0*
_class
loc:@embedding
M
lm/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
T
lm/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
T
lm/rnn/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
r
lm/rnn/rangeRangelm/rnn/range/startlm/rnn/Ranklm/rnn/range/delta*

Tidx0*
_output_shapes
:
g
lm/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
T
lm/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/concatConcatV2lm/rnn/concat/values_0lm/rnn/rangelm/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
lm/rnn/transpose	Transposelm/embedding_lookuplm/rnn/concat*
T0*,
_output_shapes
:���������@�*
Tperm0
\
lm/rnn/ShapeShapelm/rnn/transpose*
T0*
out_type0*
_output_shapes
:
d
lm/rnn/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
lm/rnn/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
f
lm/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
lm/rnn/strided_sliceStridedSlicelm/rnn/Shapelm/rnn/strided_slice/stacklm/rnn/strided_slice/stack_1lm/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
V
lm/rnn/ConstConst*
valueB:@*
dtype0*
_output_shapes
:
Y
lm/rnn/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
V
lm/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/concat_1ConcatV2lm/rnn/Constlm/rnn/Const_1lm/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
W
lm/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
lm/rnn/zerosFilllm/rnn/concat_1lm/rnn/zeros/Const*
T0*

index_type0*
_output_shapes
:	@�
M
lm/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/TensorArrayTensorArrayV3lm/rnn/strided_slice*
element_shape:	@�*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*2
tensor_array_namelm/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
�
lm/rnn/TensorArray_1TensorArrayV3lm/rnn/strided_slice*
element_shape:	@�*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*1
tensor_array_namelm/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
o
lm/rnn/TensorArrayUnstack/ShapeShapelm/rnn/transpose*
_output_shapes
:*
T0*
out_type0
w
-lm/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/lm/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/lm/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
'lm/rnn/TensorArrayUnstack/strided_sliceStridedSlicelm/rnn/TensorArrayUnstack/Shape-lm/rnn/TensorArrayUnstack/strided_slice/stack/lm/rnn/TensorArrayUnstack/strided_slice/stack_1/lm/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
g
%lm/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%lm/rnn/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
�
lm/rnn/TensorArrayUnstack/rangeRange%lm/rnn/TensorArrayUnstack/range/start'lm/rnn/TensorArrayUnstack/strided_slice%lm/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Alm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lm/rnn/TensorArray_1lm/rnn/TensorArrayUnstack/rangelm/rnn/transposelm/rnn/TensorArray_1:1*
T0*#
_class
loc:@lm/rnn/transpose*
_output_shapes
: 
R
lm/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
b
lm/rnn/MaximumMaximumlm/rnn/Maximum/xlm/rnn/strided_slice*
_output_shapes
: *
T0
`
lm/rnn/MinimumMinimumlm/rnn/strided_slicelm/rnn/Maximum*
T0*
_output_shapes
: 
`
lm/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/while/EnterEnterlm/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_1Enterlm/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_2Enterlm/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_3Enter;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�**

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_4Enter=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�**

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_5Enter=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros*
_output_shapes
:	@�**

frame_namelm/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
�
lm/rnn/while/Enter_6Enter?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1*
parallel_iterations *
_output_shapes
:	@�**

frame_namelm/rnn/while/while_context*
T0*
is_constant( 
w
lm/rnn/while/MergeMergelm/rnn/while/Enterlm/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
}
lm/rnn/while/Merge_1Mergelm/rnn/while/Enter_1lm/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
}
lm/rnn/while/Merge_2Mergelm/rnn/while/Enter_2lm/rnn/while/NextIteration_2*
_output_shapes
: : *
T0*
N
�
lm/rnn/while/Merge_3Mergelm/rnn/while/Enter_3lm/rnn/while/NextIteration_3*
T0*
N*!
_output_shapes
:	@�: 
�
lm/rnn/while/Merge_4Mergelm/rnn/while/Enter_4lm/rnn/while/NextIteration_4*
T0*
N*!
_output_shapes
:	@�: 
�
lm/rnn/while/Merge_5Mergelm/rnn/while/Enter_5lm/rnn/while/NextIteration_5*!
_output_shapes
:	@�: *
T0*
N
�
lm/rnn/while/Merge_6Mergelm/rnn/while/Enter_6lm/rnn/while/NextIteration_6*
T0*
N*!
_output_shapes
:	@�: 
g
lm/rnn/while/LessLesslm/rnn/while/Mergelm/rnn/while/Less/Enter*
_output_shapes
: *
T0
�
lm/rnn/while/Less/EnterEnterlm/rnn/strided_slice*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
m
lm/rnn/while/Less_1Lesslm/rnn/while/Merge_1lm/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
lm/rnn/while/Less_1/EnterEnterlm/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
e
lm/rnn/while/LogicalAnd
LogicalAndlm/rnn/while/Lesslm/rnn/while/Less_1*
_output_shapes
: 
R
lm/rnn/while/LoopCondLoopCondlm/rnn/while/LogicalAnd*
_output_shapes
: 
�
lm/rnn/while/SwitchSwitchlm/rnn/while/Mergelm/rnn/while/LoopCond*
T0*%
_class
loc:@lm/rnn/while/Merge*
_output_shapes
: : 
�
lm/rnn/while/Switch_1Switchlm/rnn/while/Merge_1lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_1*
_output_shapes
: : 
�
lm/rnn/while/Switch_2Switchlm/rnn/while/Merge_2lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_2*
_output_shapes
: : 
�
lm/rnn/while/Switch_3Switchlm/rnn/while/Merge_3lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_3**
_output_shapes
:	@�:	@�
�
lm/rnn/while/Switch_4Switchlm/rnn/while/Merge_4lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_4**
_output_shapes
:	@�:	@�
�
lm/rnn/while/Switch_5Switchlm/rnn/while/Merge_5lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_5**
_output_shapes
:	@�:	@�
�
lm/rnn/while/Switch_6Switchlm/rnn/while/Merge_6lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_6**
_output_shapes
:	@�:	@�
Y
lm/rnn/while/IdentityIdentitylm/rnn/while/Switch:1*
T0*
_output_shapes
: 
]
lm/rnn/while/Identity_1Identitylm/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
]
lm/rnn/while/Identity_2Identitylm/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
f
lm/rnn/while/Identity_3Identitylm/rnn/while/Switch_3:1*
T0*
_output_shapes
:	@�
f
lm/rnn/while/Identity_4Identitylm/rnn/while/Switch_4:1*
T0*
_output_shapes
:	@�
f
lm/rnn/while/Identity_5Identitylm/rnn/while/Switch_5:1*
T0*
_output_shapes
:	@�
f
lm/rnn/while/Identity_6Identitylm/rnn/while/Switch_6:1*
_output_shapes
:	@�*
T0
l
lm/rnn/while/add/yConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
lm/rnn/while/addAddlm/rnn/while/Identitylm/rnn/while/add/y*
T0*
_output_shapes
: 
�
lm/rnn/while/TensorArrayReadV3TensorArrayReadV3$lm/rnn/while/TensorArrayReadV3/Enterlm/rnn/while/Identity_1&lm/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	@�
�
$lm/rnn/while/TensorArrayReadV3/EnterEnterlm/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
&lm/rnn/while/TensorArrayReadV3/Enter_1EnterAlm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
Qrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
��*

seed *
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: 
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��
�
Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��
�
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container 
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
��
�
@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:�
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(
�
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
_output_shapes	
:�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/ConstConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axisConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concatConcatV2lm/rnn/while/TensorArrayReadV3lm/rnn/while/Identity_4Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axis*
_output_shapes
:	@�*

Tidx0*
T0*
N
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMulMatMul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concatClm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes
:	@�*
transpose_a( *
transpose_b( 
�
Clm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
��**

frame_namelm/rnn/while/while_context
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAddBiasAdd=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMulDlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	@�*
T0
�
Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:�**

frame_namelm/rnn/while/while_context
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_1Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/splitSplit<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd*@
_output_shapes.
,:	@�:	@�:	@�:	@�*
	num_split*
T0
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2Const^lm/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/AddAdd>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:2>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2*
_output_shapes
:	@�*
T0
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/SigmoidSigmoid:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add*
T0*
_output_shapes
:	@�
�
:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MulMullm/rnn/while/Identity_3>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes
:	@�*
T0
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split*
_output_shapes
:	@�*
T0
�
;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/TanhTanh>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:1*
_output_shapes
:	@�*
T0
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1Mul@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1Add:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1*
_output_shapes
:	@�*
T0
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1Tanh<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1*
_output_shapes
:	@�*
T0
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2Sigmoid>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:3*
_output_shapes
:	@�*
T0
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2Mul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	@�
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3Const^lm/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axisConst^lm/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1ConcatV2<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2lm/rnn/while/Identity_6Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axis*
T0*
N*
_output_shapes
:	@�*

Tidx0
�
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1MatMul?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1Clm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes
:	@�*
transpose_a( *
transpose_b( 
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1BiasAdd?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	@�
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_4Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1Split>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1*@
_output_shapes.
,:	@�:	@�:	@�:	@�*
	num_split*
T0
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5Const^lm/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2Add@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:2>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5*
T0*
_output_shapes
:	@�
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3Sigmoid<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2*
_output_shapes
:	@�*
T0
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3Mullm/rnn/while/Identity_5@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*
T0*
_output_shapes
:	@�
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4Sigmoid>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1*
T0*
_output_shapes
:	@�
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2Tanh@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:1*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4Mul@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*
_output_shapes
:	@�*
T0
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3Add<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4*
_output_shapes
:	@�*
T0
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3Tanh<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3*
_output_shapes
:	@�*
T0
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5Sigmoid@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:3*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5Mul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*
T0*
_output_shapes
:	@�
�
0lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterlm/rnn/while/Identity_1<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5lm/rnn/while/Identity_2*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
_output_shapes
: 
�
6lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlm/rnn/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5
n
lm/rnn/while/add_1/yConst^lm/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
i
lm/rnn/while/add_1Addlm/rnn/while/Identity_1lm/rnn/while/add_1/y*
T0*
_output_shapes
: 
^
lm/rnn/while/NextIterationNextIterationlm/rnn/while/add*
T0*
_output_shapes
: 
b
lm/rnn/while/NextIteration_1NextIterationlm/rnn/while/add_1*
_output_shapes
: *
T0
�
lm/rnn/while/NextIteration_2NextIteration0lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
lm/rnn/while/NextIteration_3NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	@�
�
lm/rnn/while/NextIteration_4NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2*
T0*
_output_shapes
:	@�
�
lm/rnn/while/NextIteration_5NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3*
T0*
_output_shapes
:	@�
�
lm/rnn/while/NextIteration_6NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
T0*
_output_shapes
:	@�
O
lm/rnn/while/ExitExitlm/rnn/while/Switch*
T0*
_output_shapes
: 
S
lm/rnn/while/Exit_1Exitlm/rnn/while/Switch_1*
T0*
_output_shapes
: 
S
lm/rnn/while/Exit_2Exitlm/rnn/while/Switch_2*
T0*
_output_shapes
: 
\
lm/rnn/while/Exit_3Exitlm/rnn/while/Switch_3*
_output_shapes
:	@�*
T0
\
lm/rnn/while/Exit_4Exitlm/rnn/while/Switch_4*
T0*
_output_shapes
:	@�
\
lm/rnn/while/Exit_5Exitlm/rnn/while/Switch_5*
_output_shapes
:	@�*
T0
\
lm/rnn/while/Exit_6Exitlm/rnn/while/Switch_6*
T0*
_output_shapes
:	@�
�
)lm/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lm/rnn/TensorArraylm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
_output_shapes
: 
�
#lm/rnn/TensorArrayStack/range/startConst*%
_class
loc:@lm/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
#lm/rnn/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *%
_class
loc:@lm/rnn/TensorArray*
value	B :
�
lm/rnn/TensorArrayStack/rangeRange#lm/rnn/TensorArrayStack/range/start)lm/rnn/TensorArrayStack/TensorArraySizeV3#lm/rnn/TensorArrayStack/range/delta*%
_class
loc:@lm/rnn/TensorArray*#
_output_shapes
:���������*

Tidx0
�
+lm/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lm/rnn/TensorArraylm/rnn/TensorArrayStack/rangelm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
dtype0*,
_output_shapes
:���������@�*
element_shape:	@�
Y
lm/rnn/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:
O
lm/rnn/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
V
lm/rnn/range_1/startConst*
_output_shapes
: *
value	B :*
dtype0
V
lm/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
z
lm/rnn/range_1Rangelm/rnn/range_1/startlm/rnn/Rank_1lm/rnn/range_1/delta*
_output_shapes
:*

Tidx0
i
lm/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
V
lm/rnn/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
lm/rnn/concat_2ConcatV2lm/rnn/concat_2/values_0lm/rnn/range_1lm/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
lm/rnn/transpose_1	Transpose+lm/rnn/TensorArrayStack/TensorArrayGatherV3lm/rnn/concat_2*,
_output_shapes
:@����������*
Tperm0*
T0
a
lm/Reshape/shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:
|

lm/ReshapeReshapelm/rnn/transpose_1lm/Reshape/shape*
Tshape0*(
_output_shapes
:����������*
T0
�
*softmax_w/Initializer/random_uniform/shapeConst*
_class
loc:@softmax_w*
valueB"�   �  *
dtype0*
_output_shapes
:
�
(softmax_w/Initializer/random_uniform/minConst*
_class
loc:@softmax_w*
valueB
 *Y��*
dtype0*
_output_shapes
: 
�
(softmax_w/Initializer/random_uniform/maxConst*
_class
loc:@softmax_w*
valueB
 *Y�<*
dtype0*
_output_shapes
: 
�
2softmax_w/Initializer/random_uniform/RandomUniformRandomUniform*softmax_w/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
��/*

seed *
T0*
_class
loc:@softmax_w
�
(softmax_w/Initializer/random_uniform/subSub(softmax_w/Initializer/random_uniform/max(softmax_w/Initializer/random_uniform/min*
_class
loc:@softmax_w*
_output_shapes
: *
T0
�
(softmax_w/Initializer/random_uniform/mulMul2softmax_w/Initializer/random_uniform/RandomUniform(softmax_w/Initializer/random_uniform/sub*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
$softmax_w/Initializer/random_uniformAdd(softmax_w/Initializer/random_uniform/mul(softmax_w/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
	softmax_w
VariableV2*
	container *
shape:
��/*
dtype0* 
_output_shapes
:
��/*
shared_name *
_class
loc:@softmax_w
�
softmax_w/AssignAssign	softmax_w$softmax_w/Initializer/random_uniform* 
_output_shapes
:
��/*
use_locking(*
T0*
_class
loc:@softmax_w*
validate_shape(
n
softmax_w/readIdentity	softmax_w*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
*softmax_b/Initializer/random_uniform/shapeConst*
_class
loc:@softmax_b*
valueB:�/*
dtype0*
_output_shapes
:
�
(softmax_b/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@softmax_b*
valueB
 *����*
dtype0
�
(softmax_b/Initializer/random_uniform/maxConst*
_class
loc:@softmax_b*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
2softmax_b/Initializer/random_uniform/RandomUniformRandomUniform*softmax_b/Initializer/random_uniform/shape*
dtype0*
_output_shapes	
:�/*

seed *
T0*
_class
loc:@softmax_b*
seed2 
�
(softmax_b/Initializer/random_uniform/subSub(softmax_b/Initializer/random_uniform/max(softmax_b/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_b*
_output_shapes
: 
�
(softmax_b/Initializer/random_uniform/mulMul2softmax_b/Initializer/random_uniform/RandomUniform(softmax_b/Initializer/random_uniform/sub*
T0*
_class
loc:@softmax_b*
_output_shapes	
:�/
�
$softmax_b/Initializer/random_uniformAdd(softmax_b/Initializer/random_uniform/mul(softmax_b/Initializer/random_uniform/min*
_class
loc:@softmax_b*
_output_shapes	
:�/*
T0
�
	softmax_b
VariableV2*
shared_name *
_class
loc:@softmax_b*
	container *
shape:�/*
dtype0*
_output_shapes	
:�/
�
softmax_b/AssignAssign	softmax_b$softmax_b/Initializer/random_uniform*
_class
loc:@softmax_b*
validate_shape(*
_output_shapes	
:�/*
use_locking(*
T0
i
softmax_b/readIdentity	softmax_b*
T0*
_class
loc:@softmax_b*
_output_shapes	
:�/
�
softmax/MatMulMatMul
lm/Reshapesoftmax_w/read*
T0*(
_output_shapes
:����������/*
transpose_a( *
transpose_b( 
�
softmax/BiasAddBiasAddsoftmax/MatMulsoftmax_b/read*
data_formatNHWC*(
_output_shapes
:����������/*
T0
`
Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
r
ReshapeReshapeinput/Placeholder_1Reshape/shape*
T0*
Tshape0*#
_output_shapes
:���������
U
one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
P
one_hot/depthConst*
dtype0*
_output_shapes
: *
value
B :�/
�
one_hotOneHotReshapeone_hot/depthone_hot/on_valueone_hot/off_value*
T0*
axis���������*
TI0*(
_output_shapes
:����������/
�
>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientone_hot*(
_output_shapes
:����������/*
T0
p
.loss/softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 
~
/loss/softmax_cross_entropy_with_logits_sg/ShapeShapesoftmax/BiasAdd*
T0*
out_type0*
_output_shapes
:
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
1loss/softmax_cross_entropy_with_logits_sg/Shape_1Shapesoftmax/BiasAdd*
T0*
out_type0*
_output_shapes
:
q
/loss/softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
-loss/softmax_cross_entropy_with_logits_sg/SubSub0loss/softmax_cross_entropy_with_logits_sg/Rank_1/loss/softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
�
5loss/softmax_cross_entropy_with_logits_sg/Slice/beginPack-loss/softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
~
4loss/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
/loss/softmax_cross_entropy_with_logits_sg/SliceSlice1loss/softmax_cross_entropy_with_logits_sg/Shape_15loss/softmax_cross_entropy_with_logits_sg/Slice/begin4loss/softmax_cross_entropy_with_logits_sg/Slice/size*
_output_shapes
:*
Index0*
T0
�
9loss/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
w
5loss/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
0loss/softmax_cross_entropy_with_logits_sg/concatConcatV29loss/softmax_cross_entropy_with_logits_sg/concat/values_0/loss/softmax_cross_entropy_with_logits_sg/Slice5loss/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
1loss/softmax_cross_entropy_with_logits_sg/ReshapeReshapesoftmax/BiasAdd0loss/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:������������������
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
�
1loss/softmax_cross_entropy_with_logits_sg/Shape_2Shape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
/loss/softmax_cross_entropy_with_logits_sg/Sub_1Sub0loss/softmax_cross_entropy_with_logits_sg/Rank_21loss/softmax_cross_entropy_with_logits_sg/Sub_1/y*
_output_shapes
: *
T0
�
7loss/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack/loss/softmax_cross_entropy_with_logits_sg/Sub_1*
_output_shapes
:*
T0*

axis *
N
�
6loss/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
1loss/softmax_cross_entropy_with_logits_sg/Slice_1Slice1loss/softmax_cross_entropy_with_logits_sg/Shape_27loss/softmax_cross_entropy_with_logits_sg/Slice_1/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_1/size*
_output_shapes
:*
Index0*
T0
�
;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
y
7loss/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
2loss/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_01loss/softmax_cross_entropy_with_logits_sg/Slice_17loss/softmax_cross_entropy_with_logits_sg/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
3loss/softmax_cross_entropy_with_logits_sg/Reshape_1Reshape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient2loss/softmax_cross_entropy_with_logits_sg/concat_1*
Tshape0*0
_output_shapes
:������������������*
T0
�
)loss/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits1loss/softmax_cross_entropy_with_logits_sg/Reshape3loss/softmax_cross_entropy_with_logits_sg/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
/loss/softmax_cross_entropy_with_logits_sg/Sub_2Sub.loss/softmax_cross_entropy_with_logits_sg/Rank1loss/softmax_cross_entropy_with_logits_sg/Sub_2/y*
_output_shapes
: *
T0
�
7loss/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
6loss/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack/loss/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
1loss/softmax_cross_entropy_with_logits_sg/Slice_2Slice/loss/softmax_cross_entropy_with_logits_sg/Shape7loss/softmax_cross_entropy_with_logits_sg/Slice_2/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
�
3loss/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape)loss/softmax_cross_entropy_with_logits_sg1loss/softmax_cross_entropy_with_logits_sg/Slice_2*#
_output_shapes
:���������*
T0*
Tshape0
T

loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
	loss/MeanMean3loss/softmax_cross_entropy_with_logits_sg/Reshape_2
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
loss_1/tagsConst*
valueB Bloss_1*
dtype0*
_output_shapes
: 
P
loss_1ScalarSummaryloss_1/tags	loss/Mean*
_output_shapes
: *
T0
�
train/gradients/ShapeShape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*#
_output_shapes
:���������
Y
train/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
train/gradients/f_count_1Entertrain/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
�
train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
_output_shapes
: : *
T0*
N
q
train/gradients/SwitchSwitchtrain/gradients/Mergelm/rnn/while/LoopCond*
T0*
_output_shapes
: : 
o
train/gradients/Add/yConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
_output_shapes
: *
T0
�
train/gradients/NextIterationNextIterationtrain/gradients/Addd^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2j^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPushV2h^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPushV2`^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
Z
train/gradients/f_count_2Exittrain/gradients/Switch*
_output_shapes
: *
T0
Y
train/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
�
train/gradients/Merge_1Mergetrain/gradients/b_count_1train/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
train/gradients/GreaterEqualGreaterEqualtrain/gradients/Merge_1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
"train/gradients/GreaterEqual/EnterEntertrain/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
[
train/gradients/b_count_2LoopCondtrain/gradients/GreaterEqual*
_output_shapes
: 
y
train/gradients/Switch_1Switchtrain/gradients/Merge_1train/gradients/b_count_2*
_output_shapes
: : *
T0
{
train/gradients/SubSubtrain/gradients/Switch_1:1"train/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
�
train/gradients/NextIteration_1NextIterationtrain/gradients/Sub_^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
\
train/gradients/b_count_3Exittrain/gradients/Switch_1*
T0*
_output_shapes
: 
�
Ntrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape)loss/softmax_cross_entropy_with_logits_sg*
out_type0*
_output_shapes
:*
T0
�
Ptrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapetrain/gradients/FillNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
train/gradients/zeros_like	ZerosLike+loss/softmax_cross_entropy_with_logits_sg:1*0
_output_shapes
:������������������*
T0
�
Mtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Itrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsPtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeMtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulMulItrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:������������������
�
Itrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax1loss/softmax_cross_entropy_with_logits_sg/Reshape*0
_output_shapes
:������������������*
T0
�
Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/NegNegItrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*0
_output_shapes
:������������������*
T0
�
Otrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Ktrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsPtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeOtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Dtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1MulKtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:������������������
�
Ltrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapesoftmax/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
Ntrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeBtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulLtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������/
�
0train/gradients/softmax/BiasAdd_grad/BiasAddGradBiasAddGradNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes	
:�/
�
*train/gradients/softmax/MatMul_grad/MatMulMatMulNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshapesoftmax_w/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/softmax/MatMul_grad/MatMul_1MatMul
lm/ReshapeNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape* 
_output_shapes
:
��/*
transpose_a(*
transpose_b( *
T0
w
%train/gradients/lm/Reshape_grad/ShapeShapelm/rnn/transpose_1*
T0*
out_type0*
_output_shapes
:
�
'train/gradients/lm/Reshape_grad/ReshapeReshape*train/gradients/softmax/MatMul_grad/MatMul%train/gradients/lm/Reshape_grad/Shape*,
_output_shapes
:@����������*
T0*
Tshape0
�
9train/gradients/lm/rnn/transpose_1_grad/InvertPermutationInvertPermutationlm/rnn/concat_2*
T0*
_output_shapes
:
�
1train/gradients/lm/rnn/transpose_1_grad/transpose	Transpose'train/gradients/lm/Reshape_grad/Reshape9train/gradients/lm/rnn/transpose_1_grad/InvertPermutation*,
_output_shapes
:���������@�*
Tperm0*
T0
�
btrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lm/rnn/TensorArraylm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
sourcetrain/gradients*
_output_shapes

:: 
�
^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylm/rnn/while/Exit_2c^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*%
_class
loc:@lm/rnn/TensorArray
�
htrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3btrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3lm/rnn/TensorArrayStack/range1train/gradients/lm/rnn/transpose_1_grad/transpose^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
v
%train/gradients/zeros/shape_as_tensorConst*
valueB"@   �   *
dtype0*
_output_shapes
:
`
train/gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zerosFill%train/gradients/zeros/shape_as_tensortrain/gradients/zeros/Const*
T0*

index_type0*
_output_shapes
:	@�
x
'train/gradients/zeros_1/shape_as_tensorConst*
_output_shapes
:*
valueB"@   �   *
dtype0
b
train/gradients/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_1Fill'train/gradients/zeros_1/shape_as_tensortrain/gradients/zeros_1/Const*
_output_shapes
:	@�*
T0*

index_type0
x
'train/gradients/zeros_2/shape_as_tensorConst*
valueB"@   �   *
dtype0*
_output_shapes
:
b
train/gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_2Fill'train/gradients/zeros_2/shape_as_tensortrain/gradients/zeros_2/Const*
_output_shapes
:	@�*
T0*

index_type0
x
'train/gradients/zeros_3/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"@   �   
b
train/gradients/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_3Fill'train/gradients/zeros_3/shape_as_tensortrain/gradients/zeros_3/Const*
_output_shapes
:	@�*
T0*

index_type0
�
/train/gradients/lm/rnn/while/Exit_2_grad/b_exitEnterhtrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
�
/train/gradients/lm/rnn/while/Exit_3_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
/train/gradients/lm/rnn/while/Exit_4_grad/b_exitEntertrain/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
/train/gradients/lm/rnn/while/Exit_5_grad/b_exitEntertrain/gradients/zeros_2*
parallel_iterations *
_output_shapes
:	@�*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant( 
�
/train/gradients/lm/rnn/while/Exit_6_grad/b_exitEntertrain/gradients/zeros_3*
is_constant( *
parallel_iterations *
_output_shapes
:	@�*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
3train/gradients/lm/rnn/while/Switch_2_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_2_grad/b_exit:train/gradients/lm/rnn/while/Switch_2_grad_1/NextIteration*
_output_shapes
: : *
T0*
N
�
3train/gradients/lm/rnn/while/Switch_3_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_3_grad/b_exit:train/gradients/lm/rnn/while/Switch_3_grad_1/NextIteration*
N*!
_output_shapes
:	@�: *
T0
�
3train/gradients/lm/rnn/while/Switch_4_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_4_grad/b_exit:train/gradients/lm/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@�: 
�
3train/gradients/lm/rnn/while/Switch_5_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_5_grad/b_exit:train/gradients/lm/rnn/while/Switch_5_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@�: 
�
3train/gradients/lm/rnn/while/Switch_6_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_6_grad/b_exit:train/gradients/lm/rnn/while/Switch_6_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@�: 
�
0train/gradients/lm/rnn/while/Merge_2_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 
�
0train/gradients/lm/rnn/while/Merge_3_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2**
_output_shapes
:	@�:	@�*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_3_grad/b_switch
�
0train/gradients/lm/rnn/while/Merge_4_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_4_grad/b_switchtrain/gradients/b_count_2*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_4_grad/b_switch**
_output_shapes
:	@�:	@�
�
0train/gradients/lm/rnn/while/Merge_5_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_5_grad/b_switchtrain/gradients/b_count_2*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_5_grad/b_switch**
_output_shapes
:	@�:	@�
�
0train/gradients/lm/rnn/while/Merge_6_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_6_grad/b_switchtrain/gradients/b_count_2*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_6_grad/b_switch**
_output_shapes
:	@�:	@�
�
.train/gradients/lm/rnn/while/Enter_2_grad/ExitExit0train/gradients/lm/rnn/while/Merge_2_grad/Switch*
T0*
_output_shapes
: 
�
.train/gradients/lm/rnn/while/Enter_3_grad/ExitExit0train/gradients/lm/rnn/while/Merge_3_grad/Switch*
_output_shapes
:	@�*
T0
�
.train/gradients/lm/rnn/while/Enter_4_grad/ExitExit0train/gradients/lm/rnn/while/Merge_4_grad/Switch*
T0*
_output_shapes
:	@�
�
.train/gradients/lm/rnn/while/Enter_5_grad/ExitExit0train/gradients/lm/rnn/while/Merge_5_grad/Switch*
_output_shapes
:	@�*
T0
�
.train/gradients/lm/rnn/while/Enter_6_grad/ExitExit0train/gradients/lm/rnn/while/Merge_6_grad/Switch*
T0*
_output_shapes
:	@�
�
gtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3mtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1*
_output_shapes

:: *O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
sourcetrain/gradients
�
mtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlm/rnn/TensorArray*
is_constant(*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
parallel_iterations 
�
ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1h^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
_output_shapes
: *
T0
�
Wtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3gtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:	@�
�
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
dtype0*
_output_shapes
: **
_class 
loc:@lm/rnn/while/Identity_1*
valueB :
���������
�
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
_output_shapes
:*
	elem_type0**
_class 
loc:@lm/rnn/while/Identity_1*

stack_name 
�
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlm/rnn/while/Identity_1^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
�
htrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerc^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2i^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2g^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2_^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
�
train/gradients/AddNAddN2train/gradients/lm/rnn/while/Merge_6_grad/Switch:1Wtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_6_grad/b_switch*
N*
_output_shapes
:	@�
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/MulMultrain/gradients/AddN`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*

stack_name *
_output_shapes
:*
	elem_type0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1Multrain/gradients/AddNbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2*
_output_shapes
:	@�*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3*

stack_name *
_output_shapes
:*
	elem_type0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@�
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3_grad/TanhGradTanhGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul*
_output_shapes
:	@�*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1*
_output_shapes
:	@�*
T0
�
:train/gradients/lm/rnn/while/Switch_2_grad_1/NextIterationNextIteration2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1*
T0*
_output_shapes
: 
�
train/gradients/AddN_1AddN2train/gradients/lm/rnn/while/Merge_5_grad/Switch:1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3_grad/TanhGrad*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_5_grad/b_switch*
N*
_output_shapes
:	@�
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/MulMultrain/gradients/AddN_1`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*

stack_name *
_output_shapes
:*
	elem_type0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1Multrain/gradients/AddN_1btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2*
_output_shapes
:	@�*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/ConstConst*
_output_shapes
: **
_class 
loc:@lm/rnn/while/Identity_5*
valueB :
���������*
dtype0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Const**
_class 
loc:@lm/rnn/while/Identity_5*

stack_name *
_output_shapes
:*
	elem_type0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Enterlm/rnn/while/Identity_5^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/MulMultrain/gradients/AddN_1`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*

stack_name *
_output_shapes
:*
	elem_type0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1Multrain/gradients/AddN_1btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@�
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4*
valueB :
���������
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1*
_output_shapes
:	@�*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4_grad/SigmoidGradSigmoidGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2_grad/TanhGradTanhGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1*
_output_shapes
:	@�*
T0
�
:train/gradients/lm/rnn/while/Switch_5_grad_1/NextIterationNextIterationUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul*
T0*
_output_shapes
:	@�
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ShapeConst^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1Const^train/gradients/Sub*
_output_shapes
: *
valueB *
dtype0
�
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ShapeYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/SumSumatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradgtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ReshapeReshapeUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/SumWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:	@�
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Sum_1Sumatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGraditrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Reshape_1ReshapeWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Sum_1Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concatConcatV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4_grad/SigmoidGrad[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2_grad/TanhGradYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Reshapeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5_grad/SigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat/Const*
T0*
N*
_output_shapes
:	@�*

Tidx0
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat/ConstConst^train/gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGradBiasAddGradZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulMatMulZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concatatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/Enter*
T0*
_output_shapes
:	@�*
transpose_a( *
transpose_b(
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1MatMulhtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/ConstConst*R
_classH
FDloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_accStackV2ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Const*R
_classH
FDloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1*

stack_name *
_output_shapes
:*
	elem_type0
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/EnterEnterctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
�
itrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPushV2StackPushV2ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Enter?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2
StackPopV2ntrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@�
�
ntrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2/EnterEnterctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConstConst^train/gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/modFloorModZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConstYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ShapeConst^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1Const^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffsetConcatOffsetXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/modZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1* 
_output_shapes
::*
N
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/SliceSlice[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffsetZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape*
_output_shapes
:	@�*
Index0*
T0
�
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice_1Slice[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffset:1\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1*
_output_shapes
:	@�*
Index0*
T0
�
train/gradients/AddN_2AddN2train/gradients/lm/rnn/while/Merge_4_grad/Switch:1Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice*
N*
_output_shapes
:	@�*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_4_grad/b_switch
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/MulMultrain/gradients/AddN_2`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
_output_shapes
:	@�*
T0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1Multrain/gradients/AddN_2btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@�
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*
_output_shapes
: *P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul*
T0*
_output_shapes
:	@�
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1*
_output_shapes
:	@�*
T0
�
:train/gradients/lm/rnn/while/Switch_6_grad_1/NextIterationNextIteration\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice_1*
T0*
_output_shapes
:	@�
�
train/gradients/AddN_3AddN2train/gradients/lm/rnn/while/Merge_3_grad/Switch:1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes
:	@�
�
Strain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/MulMultrain/gradients/AddN_3^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
_output_shapes
:	@�*
T0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*
valueB :
���������
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Const*Q
_classG
ECloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/EnterEnterYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Enter>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2dtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
dtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1Multrain/gradients/AddN_3`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/ConstConst**
_class 
loc:@lm/rnn/while/Identity_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Const**
_class 
loc:@lm/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Enterlm/rnn/while/Identity_3^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/MulMultrain/gradients/AddN_3`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
_output_shapes
:	@�*
T0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*
valueB :
���������
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Const*
	elem_type0*N
_classD
B@loc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Enter;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@�
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1Multrain/gradients/AddN_3btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
_output_shapes
:	@�*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@�
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1*
T0*
_output_shapes
:	@�
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul*
T0*
_output_shapes
:	@�
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1*
T0*
_output_shapes
:	@�
�
:train/gradients/lm/rnn/while/Switch_3_grad_1/NextIterationNextIterationStrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul*
_output_shapes
:	@�*
T0
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ShapeConst^train/gradients/Sub*
_output_shapes
:*
valueB"@   �   *
dtype0
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ShapeWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Strain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/SumSum_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradetrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ReshapeReshapeStrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/SumUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape*
_output_shapes
:	@�*
T0*
Tshape0
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Sum_1Sum_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradgtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Reshape_1ReshapeUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Sum_1Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concatConcatV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_grad/TanhGradWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Reshapeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat/Const*
N*
_output_shapes
:	@�*

Tidx0*
T0
�
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMulXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concatatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/Enter*
_output_shapes
:	@�*
transpose_a( *
transpose_b(*
T0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMulftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat*
valueB :
���������*
dtype0*
_output_shapes
: 
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnteratrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2ltrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ltrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnteratrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
train/gradients/AddN_4AddNatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad*
N*
_output_shapes	
:�*
T0
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
is_constant( *
parallel_iterations *
_output_shapes	
:�*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Mergeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:�: 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*"
_output_shapes
:�:�*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAddbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1train/gradients/AddN_4*
_output_shapes	
:�*
T0
�
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:�*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:�*
T0
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Vtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/modFloorModXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConstWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ShapeConst^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1Const^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetVtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/modXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ShapeZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/SliceSliceYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape*
_output_shapes
:	@�*
Index0*
T0
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Slice_1SliceYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1*
_output_shapes
:	@�*
Index0*
T0
�
train/gradients/AddN_5AddN]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1*
T0*p
_classf
dbloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1*
N* 
_output_shapes
:
��
�
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
��*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
��: *
T0
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*,
_output_shapes
:
��:
��
�
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAddatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1train/gradients/AddN_5*
T0* 
_output_shapes
:
��
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIteration\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
��
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
��
�
Utrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3[train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^train/gradients/Sub*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
sourcetrain/gradients*
_output_shapes

:: 
�
[train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlm/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Qtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1V^train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
�
Wtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Utrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/SliceQtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
�
Atrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterAtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant( 
�
Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Itrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
_output_shapes
: : *
T0
�
Btrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2train/gradients/b_count_2*
_output_shapes
: : *
T0
�
?train/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddDtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Wtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
�
Itrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration?train/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0
�
Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitBtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0
�
:train/gradients/lm/rnn/while/Switch_4_grad_1/NextIterationNextIterationZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Slice_1*
T0*
_output_shapes
:	@�
�
xtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lm/rnn/TensorArray_1Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*'
_class
loc:@lm/rnn/TensorArray_1*
sourcetrain/gradients*
_output_shapes

:: 
�
ttrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3y^train/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*'
_class
loc:@lm/rnn/TensorArray_1
�
jtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3xtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lm/rnn/TensorArrayUnstack/rangettrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*,
_output_shapes
:���������@�*
element_shape:*
dtype0
�
7train/gradients/lm/rnn/transpose_grad/InvertPermutationInvertPermutationlm/rnn/concat*
_output_shapes
:*
T0
�
/train/gradients/lm/rnn/transpose_grad/transpose	Transposejtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV37train/gradients/lm/rnn/transpose_grad/InvertPermutation*
T0*,
_output_shapes
:@����������*
Tperm0
�
.train/gradients/lm/embedding_lookup_grad/ShapeConst*
_output_shapes
:*
_class
loc:@embedding*%
valueB	"�      �       *
dtype0	
�
0train/gradients/lm/embedding_lookup_grad/ToInt32Cast.train/gradients/lm/embedding_lookup_grad/Shape*

SrcT0	*
_class
loc:@embedding*
_output_shapes
:*

DstT0
y
-train/gradients/lm/embedding_lookup_grad/SizeSizeinput/Placeholder*
T0*
out_type0*
_output_shapes
: 
y
7train/gradients/lm/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
3train/gradients/lm/embedding_lookup_grad/ExpandDims
ExpandDims-train/gradients/lm/embedding_lookup_grad/Size7train/gradients/lm/embedding_lookup_grad/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
�
<train/gradients/lm/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
�
>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/lm/embedding_lookup_grad/strided_sliceStridedSlice0train/gradients/lm/embedding_lookup_grad/ToInt32<train/gradients/lm/embedding_lookup_grad/strided_slice/stack>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_1>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_2*
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask 
v
4train/gradients/lm/embedding_lookup_grad/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
/train/gradients/lm/embedding_lookup_grad/concatConcatV23train/gradients/lm/embedding_lookup_grad/ExpandDims6train/gradients/lm/embedding_lookup_grad/strided_slice4train/gradients/lm/embedding_lookup_grad/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
0train/gradients/lm/embedding_lookup_grad/ReshapeReshape/train/gradients/lm/rnn/transpose_grad/transpose/train/gradients/lm/embedding_lookup_grad/concat*
T0*
Tshape0*(
_output_shapes
:����������
�
2train/gradients/lm/embedding_lookup_grad/Reshape_1Reshapeinput/Placeholder3train/gradients/lm/embedding_lookup_grad/ExpandDims*#
_output_shapes
:���������*
T0*
Tshape0
�
train/global_norm/L2LossL2Loss0train/gradients/lm/embedding_lookup_grad/Reshape*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape*
_output_shapes
: 
�
train/global_norm/L2Loss_1L2Loss`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3
�
train/global_norm/L2Loss_2L2Lossatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: 
�
train/global_norm/L2Loss_3L2Loss,train/gradients/softmax/MatMul_grad/MatMul_1*
_output_shapes
: *
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1
�
train/global_norm/L2Loss_4L2Loss0train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
train/global_norm/stackPacktrain/global_norm/L2Losstrain/global_norm/L2Loss_1train/global_norm/L2Loss_2train/global_norm/L2Loss_3train/global_norm/L2Loss_4*
_output_shapes
:*
T0*

axis *
N
a
train/global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
train/global_norm/SumSumtrain/global_norm/stacktrain/global_norm/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
^
train/global_norm/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   @
o
train/global_norm/mulMultrain/global_norm/Sumtrain/global_norm/Const_1*
T0*
_output_shapes
: 
]
train/global_norm/global_normSqrttrain/global_norm/mul*
T0*
_output_shapes
: 
h
#train/clip_by_global_norm/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!train/clip_by_global_norm/truedivRealDiv#train/clip_by_global_norm/truediv/xtrain/global_norm/global_norm*
_output_shapes
: *
T0
d
train/clip_by_global_norm/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
j
%train/clip_by_global_norm/truediv_1/yConst*
valueB
 *  �@*
dtype0*
_output_shapes
: 
�
#train/clip_by_global_norm/truediv_1RealDivtrain/clip_by_global_norm/Const%train/clip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 
�
!train/clip_by_global_norm/MinimumMinimum!train/clip_by_global_norm/truediv#train/clip_by_global_norm/truediv_1*
_output_shapes
: *
T0
d
train/clip_by_global_norm/mul/xConst*
valueB
 *  �@*
dtype0*
_output_shapes
: 
�
train/clip_by_global_norm/mulMultrain/clip_by_global_norm/mul/x!train/clip_by_global_norm/Minimum*
T0*
_output_shapes
: 
�
train/clip_by_global_norm/mul_1Mul0train/gradients/lm/embedding_lookup_grad/Reshapetrain/clip_by_global_norm/mul*(
_output_shapes
:����������*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape
�
6train/clip_by_global_norm/train/clip_by_global_norm/_0Identitytrain/clip_by_global_norm/mul_1*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape*(
_output_shapes
:����������*
T0
�
train/clip_by_global_norm/mul_2Mul`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3train/clip_by_global_norm/mul*
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��
�
6train/clip_by_global_norm/train/clip_by_global_norm/_1Identitytrain/clip_by_global_norm/mul_2*
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��
�
train/clip_by_global_norm/mul_3Mulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3train/clip_by_global_norm/mul*
_output_shapes	
:�*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
�
6train/clip_by_global_norm/train/clip_by_global_norm/_2Identitytrain/clip_by_global_norm/mul_3*
_output_shapes	
:�*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
�
train/clip_by_global_norm/mul_4Mul,train/gradients/softmax/MatMul_grad/MatMul_1train/clip_by_global_norm/mul*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1* 
_output_shapes
:
��/
�
6train/clip_by_global_norm/train/clip_by_global_norm/_3Identitytrain/clip_by_global_norm/mul_4*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1* 
_output_shapes
:
��/
�
train/clip_by_global_norm/mul_5Mul0train/gradients/softmax/BiasAdd_grad/BiasAddGradtrain/clip_by_global_norm/mul*
_output_shapes	
:�/*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad
�
6train/clip_by_global_norm/train/clip_by_global_norm/_4Identitytrain/clip_by_global_norm/mul_5*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�/
�
train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *fff?
�
train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@embedding*
	container *
shape: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
t
train/beta1_power/readIdentitytrain/beta1_power*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
_class
loc:@embedding*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@embedding
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
t
train/beta2_power/readIdentitytrain/beta2_power*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
0embedding/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@embedding*
valueB"�  �   *
dtype0*
_output_shapes
:
�
&embedding/Adam/Initializer/zeros/ConstConst*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 embedding/Adam/Initializer/zerosFill0embedding/Adam/Initializer/zeros/shape_as_tensor&embedding/Adam/Initializer/zeros/Const*
_class
loc:@embedding*

index_type0* 
_output_shapes
:
�/�*
T0
�
embedding/Adam
VariableV2*
shared_name *
_class
loc:@embedding*
	container *
shape:
�/�*
dtype0* 
_output_shapes
:
�/�
�
embedding/Adam/AssignAssignembedding/Adam embedding/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�
x
embedding/Adam/readIdentityembedding/Adam*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
2embedding/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@embedding*
valueB"�  �   *
dtype0*
_output_shapes
:
�
(embedding/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *    
�
"embedding/Adam_1/Initializer/zerosFill2embedding/Adam_1/Initializer/zeros/shape_as_tensor(embedding/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
�/�*
T0*
_class
loc:@embedding*

index_type0
�
embedding/Adam_1
VariableV2*
shape:
�/�*
dtype0* 
_output_shapes
:
�/�*
shared_name *
_class
loc:@embedding*
	container 
�
embedding/Adam_1/AssignAssignembedding/Adam_1"embedding/Adam_1/Initializer/zeros*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�*
use_locking(*
T0
|
embedding/Adam_1/readIdentityembedding/Adam_1*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
Wrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      
�
Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zerosFillWrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorMrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*

index_type0
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��
�
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*

index_type0* 
_output_shapes
:
��
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
��
�
>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��
�
Ernn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:�
�
8rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:�
�
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container 
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:�
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:�
�
0softmax_w/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@softmax_w*
valueB"�   �  *
dtype0
�
&softmax_w/Adam/Initializer/zeros/ConstConst*
_class
loc:@softmax_w*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 softmax_w/Adam/Initializer/zerosFill0softmax_w/Adam/Initializer/zeros/shape_as_tensor&softmax_w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@softmax_w*

index_type0* 
_output_shapes
:
��/
�
softmax_w/Adam
VariableV2*
	container *
shape:
��/*
dtype0* 
_output_shapes
:
��/*
shared_name *
_class
loc:@softmax_w
�
softmax_w/Adam/AssignAssignsoftmax_w/Adam softmax_w/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@softmax_w*
validate_shape(* 
_output_shapes
:
��/
x
softmax_w/Adam/readIdentitysoftmax_w/Adam*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
2softmax_w/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@softmax_w*
valueB"�   �  
�
(softmax_w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@softmax_w*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_w/Adam_1/Initializer/zerosFill2softmax_w/Adam_1/Initializer/zeros/shape_as_tensor(softmax_w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@softmax_w*

index_type0* 
_output_shapes
:
��/
�
softmax_w/Adam_1
VariableV2*
shape:
��/*
dtype0* 
_output_shapes
:
��/*
shared_name *
_class
loc:@softmax_w*
	container 
�
softmax_w/Adam_1/AssignAssignsoftmax_w/Adam_1"softmax_w/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��/*
use_locking(*
T0*
_class
loc:@softmax_w
|
softmax_w/Adam_1/readIdentitysoftmax_w/Adam_1*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
0softmax_b/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_b*
valueB:�/*
dtype0*
_output_shapes
:
�
&softmax_b/Adam/Initializer/zeros/ConstConst*
_class
loc:@softmax_b*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 softmax_b/Adam/Initializer/zerosFill0softmax_b/Adam/Initializer/zeros/shape_as_tensor&softmax_b/Adam/Initializer/zeros/Const*
_output_shapes	
:�/*
T0*
_class
loc:@softmax_b*

index_type0
�
softmax_b/Adam
VariableV2*
dtype0*
_output_shapes	
:�/*
shared_name *
_class
loc:@softmax_b*
	container *
shape:�/
�
softmax_b/Adam/AssignAssignsoftmax_b/Adam softmax_b/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@softmax_b*
validate_shape(*
_output_shapes	
:�/
s
softmax_b/Adam/readIdentitysoftmax_b/Adam*
T0*
_class
loc:@softmax_b*
_output_shapes	
:�/
�
2softmax_b/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@softmax_b*
valueB:�/*
dtype0
�
(softmax_b/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@softmax_b*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_b/Adam_1/Initializer/zerosFill2softmax_b/Adam_1/Initializer/zeros/shape_as_tensor(softmax_b/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�/*
T0*
_class
loc:@softmax_b*

index_type0
�
softmax_b/Adam_1
VariableV2*
shape:�/*
dtype0*
_output_shapes	
:�/*
shared_name *
_class
loc:@softmax_b*
	container 
�
softmax_b/Adam_1/AssignAssignsoftmax_b/Adam_1"softmax_b/Adam_1/Initializer/zeros*
_output_shapes	
:�/*
use_locking(*
T0*
_class
loc:@softmax_b*
validate_shape(
w
softmax_b/Adam_1/readIdentitysoftmax_b/Adam_1*
T0*
_class
loc:@softmax_b*
_output_shapes	
:�/
U
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
"train/Adam/update_embedding/UniqueUnique2train/gradients/lm/embedding_lookup_grad/Reshape_1*
out_idx0*
T0*
_class
loc:@embedding*2
_output_shapes 
:���������:���������
�
!train/Adam/update_embedding/ShapeShape"train/Adam/update_embedding/Unique*
_class
loc:@embedding*
out_type0*
_output_shapes
:*
T0
�
/train/Adam/update_embedding/strided_slice/stackConst*
_class
loc:@embedding*
valueB: *
dtype0*
_output_shapes
:
�
1train/Adam/update_embedding/strided_slice/stack_1Const*
_class
loc:@embedding*
valueB:*
dtype0*
_output_shapes
:
�
1train/Adam/update_embedding/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
_class
loc:@embedding*
valueB:
�
)train/Adam/update_embedding/strided_sliceStridedSlice!train/Adam/update_embedding/Shape/train/Adam/update_embedding/strided_slice/stack1train/Adam/update_embedding/strided_slice/stack_11train/Adam/update_embedding/strided_slice/stack_2*
Index0*
T0*
_class
loc:@embedding*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
.train/Adam/update_embedding/UnsortedSegmentSumUnsortedSegmentSum6train/clip_by_global_norm/train/clip_by_global_norm/_0$train/Adam/update_embedding/Unique:1)train/Adam/update_embedding/strided_slice*
_class
loc:@embedding*(
_output_shapes
:����������*
Tnumsegments0*
Tindices0*
T0
�
!train/Adam/update_embedding/sub/xConst*
_class
loc:@embedding*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/Adam/update_embedding/subSub!train/Adam/update_embedding/sub/xtrain/beta2_power/read*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
 train/Adam/update_embedding/SqrtSqrttrain/Adam/update_embedding/sub*
_class
loc:@embedding*
_output_shapes
: *
T0
�
train/Adam/update_embedding/mulMullearning_rate/Variable/read train/Adam/update_embedding/Sqrt*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
#train/Adam/update_embedding/sub_1/xConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *  �?
�
!train/Adam/update_embedding/sub_1Sub#train/Adam/update_embedding/sub_1/xtrain/beta1_power/read*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
#train/Adam/update_embedding/truedivRealDivtrain/Adam/update_embedding/mul!train/Adam/update_embedding/sub_1*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
#train/Adam/update_embedding/sub_2/xConst*
_class
loc:@embedding*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!train/Adam/update_embedding/sub_2Sub#train/Adam/update_embedding/sub_2/xtrain/Adam/beta1*
_class
loc:@embedding*
_output_shapes
: *
T0
�
!train/Adam/update_embedding/mul_1Mul.train/Adam/update_embedding/UnsortedSegmentSum!train/Adam/update_embedding/sub_2*
T0*
_class
loc:@embedding*(
_output_shapes
:����������
�
!train/Adam/update_embedding/mul_2Mulembedding/Adam/readtrain/Adam/beta1*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
"train/Adam/update_embedding/AssignAssignembedding/Adam!train/Adam/update_embedding/mul_2*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�*
use_locking( 
�
&train/Adam/update_embedding/ScatterAdd
ScatterAddembedding/Adam"train/Adam/update_embedding/Unique!train/Adam/update_embedding/mul_1#^train/Adam/update_embedding/Assign*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
!train/Adam/update_embedding/mul_3Mul.train/Adam/update_embedding/UnsortedSegmentSum.train/Adam/update_embedding/UnsortedSegmentSum*(
_output_shapes
:����������*
T0*
_class
loc:@embedding
�
#train/Adam/update_embedding/sub_3/xConst*
_class
loc:@embedding*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!train/Adam/update_embedding/sub_3Sub#train/Adam/update_embedding/sub_3/xtrain/Adam/beta2*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
!train/Adam/update_embedding/mul_4Mul!train/Adam/update_embedding/mul_3!train/Adam/update_embedding/sub_3*
T0*
_class
loc:@embedding*(
_output_shapes
:����������
�
!train/Adam/update_embedding/mul_5Mulembedding/Adam_1/readtrain/Adam/beta2*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
$train/Adam/update_embedding/Assign_1Assignembedding/Adam_1!train/Adam/update_embedding/mul_5*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�*
use_locking( *
T0
�
(train/Adam/update_embedding/ScatterAdd_1
ScatterAddembedding/Adam_1"train/Adam/update_embedding/Unique!train/Adam/update_embedding/mul_4%^train/Adam/update_embedding/Assign_1*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
"train/Adam/update_embedding/Sqrt_1Sqrt(train/Adam/update_embedding/ScatterAdd_1*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
!train/Adam/update_embedding/mul_6Mul#train/Adam/update_embedding/truediv&train/Adam/update_embedding/ScatterAdd* 
_output_shapes
:
�/�*
T0*
_class
loc:@embedding
�
train/Adam/update_embedding/addAdd"train/Adam/update_embedding/Sqrt_1train/Adam/epsilon*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
%train/Adam/update_embedding/truediv_1RealDiv!train/Adam/update_embedding/mul_6train/Adam/update_embedding/add*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
%train/Adam/update_embedding/AssignSub	AssignSub	embedding%train/Adam/update_embedding/truediv_1* 
_output_shapes
:
�/�*
use_locking( *
T0*
_class
loc:@embedding
�
&train/Adam/update_embedding/group_depsNoOp&^train/Adam/update_embedding/AssignSub'^train/Adam/update_embedding/ScatterAdd)^train/Adam/update_embedding/ScatterAdd_1*
_class
loc:@embedding
�
Ltrain/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
Jtrain/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_2*
use_locking( *
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes	
:�
�
%train/Adam/update_softmax_w/ApplyAdam	ApplyAdam	softmax_wsoftmax_w/Adamsoftmax_w/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_3* 
_output_shapes
:
��/*
use_locking( *
T0*
_class
loc:@softmax_w*
use_nesterov( 
�
%train/Adam/update_softmax_b/ApplyAdam	ApplyAdam	softmax_bsoftmax_b/Adamsoftmax_b/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_4*
use_locking( *
T0*
_class
loc:@softmax_b*
use_nesterov( *
_output_shapes	
:�/
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam
�
initNoOp^embedding/Adam/Assign^embedding/Adam_1/Assign^embedding/Assign^learning_rate/Variable/Assign;^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign6^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign8^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign^softmax_b/Adam/Assign^softmax_b/Adam_1/Assign^softmax_b/Assign^softmax_w/Adam/Assign^softmax_w/Adam_1/Assign^softmax_w/Assign^train/beta1_power/Assign^train/beta2_power/Assign
\
Merge/MergeSummaryMergeSummarylearning_rate_1loss_1*
N*
_output_shapes
: "�2[�<�     )���	������AJ��
�5�4
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignSub
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
2
L2Loss
t"T
output"T"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
$

LogicalAnd
x

y

z
�
!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
�

ScatterAdd
ref"T�
indices"Tindices
updates"T

output_ref"T�" 
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
-
Sqrt
x"T
y"T"
Ttype:

2
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype�
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( �
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring �
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:�
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring�
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype�
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype�
9
TensorArraySizeV3

handle
flow_in
size�
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring �
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype�
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
�
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.8.02v1.8.0-0-g93bc2e2072��	
t
input/PlaceholderPlaceholder*
shape:@���������*
dtype0*'
_output_shapes
:@���������
v
input/Placeholder_1Placeholder*
shape:@���������*
dtype0*'
_output_shapes
:@���������
i
$learning_rate/Variable/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
learning_rate/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
learning_rate/Variable/AssignAssignlearning_rate/Variable$learning_rate/Variable/initial_value*
use_locking(*
T0*)
_class
loc:@learning_rate/Variable*
validate_shape(*
_output_shapes
: 
�
learning_rate/Variable/readIdentitylearning_rate/Variable*
_output_shapes
: *
T0*)
_class
loc:@learning_rate/Variable
d
learning_rate_1/tagsConst* 
valueB Blearning_rate_1*
dtype0*
_output_shapes
: 
t
learning_rate_1ScalarSummarylearning_rate_1/tagslearning_rate/Variable/read*
T0*
_output_shapes
: 
�
;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:@*
dtype0*
_output_shapes
:
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
<rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concatConcatV2;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_1Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zerosFill<rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concatArnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes
:	@�
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:@*
dtype0*
_output_shapes
:
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:�*
dtype0*
_output_shapes
:
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:@*
dtype0*
_output_shapes
:
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:�*
dtype0*
_output_shapes
:
�
Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1ConcatV2=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_4=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_5Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1Fill>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes
:	@�*
T0*

index_type0
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:@
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
valueB:�*
dtype0
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ConstConst*
valueB:@*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatConcatV2=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_1Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zerosFill>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatCrnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/Const*
T0*

index_type0*
_output_shapes
:	@�
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_2Const*
valueB:@*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_3Const*
valueB:�*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_4Const*
valueB:@*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_5Const*
dtype0*
_output_shapes
:*
valueB:�
�
Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
@rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1ConcatV2?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_4?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_5Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1Fill@rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	@�
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_6Const*
valueB:@*
dtype0*
_output_shapes
:
�
?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_7Const*
valueB:�*
dtype0*
_output_shapes
:
�
*embedding/Initializer/random_uniform/shapeConst*
_class
loc:@embedding*
valueB"�  �   *
dtype0*
_output_shapes
:
�
(embedding/Initializer/random_uniform/minConst*
_class
loc:@embedding*
valueB
 *Y��*
dtype0*
_output_shapes
: 
�
(embedding/Initializer/random_uniform/maxConst*
_class
loc:@embedding*
valueB
 *Y�<*
dtype0*
_output_shapes
: 
�
2embedding/Initializer/random_uniform/RandomUniformRandomUniform*embedding/Initializer/random_uniform/shape*
_class
loc:@embedding*
seed2 *
dtype0* 
_output_shapes
:
�/�*

seed *
T0
�
(embedding/Initializer/random_uniform/subSub(embedding/Initializer/random_uniform/max(embedding/Initializer/random_uniform/min*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
(embedding/Initializer/random_uniform/mulMul2embedding/Initializer/random_uniform/RandomUniform(embedding/Initializer/random_uniform/sub*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
$embedding/Initializer/random_uniformAdd(embedding/Initializer/random_uniform/mul(embedding/Initializer/random_uniform/min*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
	embedding
VariableV2* 
_output_shapes
:
�/�*
shared_name *
_class
loc:@embedding*
	container *
shape:
�/�*
dtype0
�
embedding/AssignAssign	embedding$embedding/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�
n
embedding/readIdentity	embedding*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
x
lm/embedding_lookup/axisConst*
_class
loc:@embedding*
value	B : *
dtype0*
_output_shapes
: 
�
lm/embedding_lookupGatherV2embedding/readinput/Placeholderlm/embedding_lookup/axis*
_class
loc:@embedding*,
_output_shapes
:@����������*
Taxis0*
Tindices0*
Tparams0
M
lm/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
T
lm/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
T
lm/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
r
lm/rnn/rangeRangelm/rnn/range/startlm/rnn/Ranklm/rnn/range/delta*
_output_shapes
:*

Tidx0
g
lm/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
T
lm/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/concatConcatV2lm/rnn/concat/values_0lm/rnn/rangelm/rnn/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
lm/rnn/transpose	Transposelm/embedding_lookuplm/rnn/concat*,
_output_shapes
:���������@�*
Tperm0*
T0
\
lm/rnn/ShapeShapelm/rnn/transpose*
_output_shapes
:*
T0*
out_type0
d
lm/rnn/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
lm/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
lm/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
lm/rnn/strided_sliceStridedSlicelm/rnn/Shapelm/rnn/strided_slice/stacklm/rnn/strided_slice/stack_1lm/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
V
lm/rnn/ConstConst*
valueB:@*
dtype0*
_output_shapes
:
Y
lm/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB:�
V
lm/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/concat_1ConcatV2lm/rnn/Constlm/rnn/Const_1lm/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
W
lm/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
lm/rnn/zerosFilllm/rnn/concat_1lm/rnn/zeros/Const*
T0*

index_type0*
_output_shapes
:	@�
M
lm/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/TensorArrayTensorArrayV3lm/rnn/strided_slice*
element_shape:	@�*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*2
tensor_array_namelm/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
�
lm/rnn/TensorArray_1TensorArrayV3lm/rnn/strided_slice*1
tensor_array_namelm/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:	@�*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
o
lm/rnn/TensorArrayUnstack/ShapeShapelm/rnn/transpose*
_output_shapes
:*
T0*
out_type0
w
-lm/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/lm/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/lm/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
'lm/rnn/TensorArrayUnstack/strided_sliceStridedSlicelm/rnn/TensorArrayUnstack/Shape-lm/rnn/TensorArrayUnstack/strided_slice/stack/lm/rnn/TensorArrayUnstack/strided_slice/stack_1/lm/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
g
%lm/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%lm/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
lm/rnn/TensorArrayUnstack/rangeRange%lm/rnn/TensorArrayUnstack/range/start'lm/rnn/TensorArrayUnstack/strided_slice%lm/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
Alm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lm/rnn/TensorArray_1lm/rnn/TensorArrayUnstack/rangelm/rnn/transposelm/rnn/TensorArray_1:1*
T0*#
_class
loc:@lm/rnn/transpose*
_output_shapes
: 
R
lm/rnn/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :
b
lm/rnn/MaximumMaximumlm/rnn/Maximum/xlm/rnn/strided_slice*
_output_shapes
: *
T0
`
lm/rnn/MinimumMinimumlm/rnn/strided_slicelm/rnn/Maximum*
T0*
_output_shapes
: 
`
lm/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/while/EnterEnterlm/rnn/while/iteration_counter*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0*
is_constant( 
�
lm/rnn/while/Enter_1Enterlm/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_2Enterlm/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_3Enter;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�**

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_4Enter=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�**

frame_namelm/rnn/while/while_context
�
lm/rnn/while/Enter_5Enter=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros*
is_constant( *
parallel_iterations *
_output_shapes
:	@�**

frame_namelm/rnn/while/while_context*
T0
�
lm/rnn/while/Enter_6Enter?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�**

frame_namelm/rnn/while/while_context
w
lm/rnn/while/MergeMergelm/rnn/while/Enterlm/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
}
lm/rnn/while/Merge_1Mergelm/rnn/while/Enter_1lm/rnn/while/NextIteration_1*
_output_shapes
: : *
T0*
N
}
lm/rnn/while/Merge_2Mergelm/rnn/while/Enter_2lm/rnn/while/NextIteration_2*
N*
_output_shapes
: : *
T0
�
lm/rnn/while/Merge_3Mergelm/rnn/while/Enter_3lm/rnn/while/NextIteration_3*!
_output_shapes
:	@�: *
T0*
N
�
lm/rnn/while/Merge_4Mergelm/rnn/while/Enter_4lm/rnn/while/NextIteration_4*
T0*
N*!
_output_shapes
:	@�: 
�
lm/rnn/while/Merge_5Mergelm/rnn/while/Enter_5lm/rnn/while/NextIteration_5*
T0*
N*!
_output_shapes
:	@�: 
�
lm/rnn/while/Merge_6Mergelm/rnn/while/Enter_6lm/rnn/while/NextIteration_6*
T0*
N*!
_output_shapes
:	@�: 
g
lm/rnn/while/LessLesslm/rnn/while/Mergelm/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
lm/rnn/while/Less/EnterEnterlm/rnn/strided_slice*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0*
is_constant(
m
lm/rnn/while/Less_1Lesslm/rnn/while/Merge_1lm/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
lm/rnn/while/Less_1/EnterEnterlm/rnn/Minimum*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0*
is_constant(
e
lm/rnn/while/LogicalAnd
LogicalAndlm/rnn/while/Lesslm/rnn/while/Less_1*
_output_shapes
: 
R
lm/rnn/while/LoopCondLoopCondlm/rnn/while/LogicalAnd*
_output_shapes
: 
�
lm/rnn/while/SwitchSwitchlm/rnn/while/Mergelm/rnn/while/LoopCond*
_output_shapes
: : *
T0*%
_class
loc:@lm/rnn/while/Merge
�
lm/rnn/while/Switch_1Switchlm/rnn/while/Merge_1lm/rnn/while/LoopCond*'
_class
loc:@lm/rnn/while/Merge_1*
_output_shapes
: : *
T0
�
lm/rnn/while/Switch_2Switchlm/rnn/while/Merge_2lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_2*
_output_shapes
: : 
�
lm/rnn/while/Switch_3Switchlm/rnn/while/Merge_3lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_3**
_output_shapes
:	@�:	@�
�
lm/rnn/while/Switch_4Switchlm/rnn/while/Merge_4lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_4**
_output_shapes
:	@�:	@�
�
lm/rnn/while/Switch_5Switchlm/rnn/while/Merge_5lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_5**
_output_shapes
:	@�:	@�
�
lm/rnn/while/Switch_6Switchlm/rnn/while/Merge_6lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_6**
_output_shapes
:	@�:	@�
Y
lm/rnn/while/IdentityIdentitylm/rnn/while/Switch:1*
_output_shapes
: *
T0
]
lm/rnn/while/Identity_1Identitylm/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
]
lm/rnn/while/Identity_2Identitylm/rnn/while/Switch_2:1*
_output_shapes
: *
T0
f
lm/rnn/while/Identity_3Identitylm/rnn/while/Switch_3:1*
T0*
_output_shapes
:	@�
f
lm/rnn/while/Identity_4Identitylm/rnn/while/Switch_4:1*
_output_shapes
:	@�*
T0
f
lm/rnn/while/Identity_5Identitylm/rnn/while/Switch_5:1*
T0*
_output_shapes
:	@�
f
lm/rnn/while/Identity_6Identitylm/rnn/while/Switch_6:1*
T0*
_output_shapes
:	@�
l
lm/rnn/while/add/yConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
lm/rnn/while/addAddlm/rnn/while/Identitylm/rnn/while/add/y*
T0*
_output_shapes
: 
�
lm/rnn/while/TensorArrayReadV3TensorArrayReadV3$lm/rnn/while/TensorArrayReadV3/Enterlm/rnn/while/Identity_1&lm/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	@�
�
$lm/rnn/while/TensorArrayReadV3/EnterEnterlm/rnn/TensorArray_1*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
�
&lm/rnn/while/TensorArrayReadV3/Enter_1EnterAlm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0
�
Qrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *���
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
seed2 *
dtype0* 
_output_shapes
:
��*

seed 
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: *
T0
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��*
T0
�
Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��
�
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
��
�
@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:�
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(
�
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
_output_shapes	
:�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/ConstConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axisConst^lm/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concatConcatV2lm/rnn/while/TensorArrayReadV3lm/rnn/while/Identity_4Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:	@�
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMulMatMul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concatClm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter*
_output_shapes
:	@�*
transpose_a( *
transpose_b( *
T0
�
Clm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
��**

frame_namelm/rnn/while/while_context
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAddBiasAdd=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMulDlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	@�
�
Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
_output_shapes	
:�**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_1Const^lm/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/splitSplit<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd*@
_output_shapes.
,:	@�:	@�:	@�:	@�*
	num_split*
T0
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2Const^lm/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/AddAdd>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:2>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2*
_output_shapes
:	@�*
T0
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/SigmoidSigmoid:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add*
T0*
_output_shapes
:	@�
�
:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MulMullm/rnn/while/Identity_3>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*
T0*
_output_shapes
:	@�
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split*
_output_shapes
:	@�*
T0
�
;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/TanhTanh>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:1*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1Mul@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*
_output_shapes
:	@�*
T0
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1Add:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	@�
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1Tanh<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1*
_output_shapes
:	@�*
T0
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2Sigmoid>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:3*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2Mul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*
_output_shapes
:	@�*
T0
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axisConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1ConcatV2<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2lm/rnn/while/Identity_6Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:	@�
�
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1MatMul?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1Clm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter*
_output_shapes
:	@�*
transpose_a( *
transpose_b( *
T0
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1BiasAdd?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	@�*
T0
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_4Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1Split>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1*@
_output_shapes.
,:	@�:	@�:	@�:	@�*
	num_split*
T0
�
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5Const^lm/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2Add@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:2>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5*
T0*
_output_shapes
:	@�
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3Sigmoid<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3Mullm/rnn/while/Identity_5@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*
T0*
_output_shapes
:	@�
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4Sigmoid>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1*
_output_shapes
:	@�*
T0
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2Tanh@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:1*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4Mul@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3Add<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4*
T0*
_output_shapes
:	@�
�
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3Tanh<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3*
_output_shapes
:	@�*
T0
�
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5Sigmoid@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:3*
T0*
_output_shapes
:	@�
�
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5Mul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*
T0*
_output_shapes
:	@�
�
0lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterlm/rnn/while/Identity_1<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5lm/rnn/while/Identity_2*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
_output_shapes
: *
T0
�
6lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlm/rnn/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5
n
lm/rnn/while/add_1/yConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
lm/rnn/while/add_1Addlm/rnn/while/Identity_1lm/rnn/while/add_1/y*
_output_shapes
: *
T0
^
lm/rnn/while/NextIterationNextIterationlm/rnn/while/add*
_output_shapes
: *
T0
b
lm/rnn/while/NextIteration_1NextIterationlm/rnn/while/add_1*
_output_shapes
: *
T0
�
lm/rnn/while/NextIteration_2NextIteration0lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
�
lm/rnn/while/NextIteration_3NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	@�
�
lm/rnn/while/NextIteration_4NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2*
_output_shapes
:	@�*
T0
�
lm/rnn/while/NextIteration_5NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3*
_output_shapes
:	@�*
T0
�
lm/rnn/while/NextIteration_6NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
_output_shapes
:	@�*
T0
O
lm/rnn/while/ExitExitlm/rnn/while/Switch*
T0*
_output_shapes
: 
S
lm/rnn/while/Exit_1Exitlm/rnn/while/Switch_1*
_output_shapes
: *
T0
S
lm/rnn/while/Exit_2Exitlm/rnn/while/Switch_2*
T0*
_output_shapes
: 
\
lm/rnn/while/Exit_3Exitlm/rnn/while/Switch_3*
T0*
_output_shapes
:	@�
\
lm/rnn/while/Exit_4Exitlm/rnn/while/Switch_4*
T0*
_output_shapes
:	@�
\
lm/rnn/while/Exit_5Exitlm/rnn/while/Switch_5*
T0*
_output_shapes
:	@�
\
lm/rnn/while/Exit_6Exitlm/rnn/while/Switch_6*
T0*
_output_shapes
:	@�
�
)lm/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lm/rnn/TensorArraylm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
_output_shapes
: 
�
#lm/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *%
_class
loc:@lm/rnn/TensorArray*
value	B : 
�
#lm/rnn/TensorArrayStack/range/deltaConst*
_output_shapes
: *%
_class
loc:@lm/rnn/TensorArray*
value	B :*
dtype0
�
lm/rnn/TensorArrayStack/rangeRange#lm/rnn/TensorArrayStack/range/start)lm/rnn/TensorArrayStack/TensorArraySizeV3#lm/rnn/TensorArrayStack/range/delta*%
_class
loc:@lm/rnn/TensorArray*#
_output_shapes
:���������*

Tidx0
�
+lm/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lm/rnn/TensorArraylm/rnn/TensorArrayStack/rangelm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
dtype0*,
_output_shapes
:���������@�*
element_shape:	@�
Y
lm/rnn/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:
O
lm/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
V
lm/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
V
lm/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
z
lm/rnn/range_1Rangelm/rnn/range_1/startlm/rnn/Rank_1lm/rnn/range_1/delta*

Tidx0*
_output_shapes
:
i
lm/rnn/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
V
lm/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lm/rnn/concat_2ConcatV2lm/rnn/concat_2/values_0lm/rnn/range_1lm/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
lm/rnn/transpose_1	Transpose+lm/rnn/TensorArrayStack/TensorArrayGatherV3lm/rnn/concat_2*
T0*,
_output_shapes
:@����������*
Tperm0
a
lm/Reshape/shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:
|

lm/ReshapeReshapelm/rnn/transpose_1lm/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
*softmax_w/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@softmax_w*
valueB"�   �  *
dtype0
�
(softmax_w/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@softmax_w*
valueB
 *Y��*
dtype0
�
(softmax_w/Initializer/random_uniform/maxConst*
_class
loc:@softmax_w*
valueB
 *Y�<*
dtype0*
_output_shapes
: 
�
2softmax_w/Initializer/random_uniform/RandomUniformRandomUniform*softmax_w/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��/*

seed *
T0*
_class
loc:@softmax_w*
seed2 
�
(softmax_w/Initializer/random_uniform/subSub(softmax_w/Initializer/random_uniform/max(softmax_w/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_w*
_output_shapes
: 
�
(softmax_w/Initializer/random_uniform/mulMul2softmax_w/Initializer/random_uniform/RandomUniform(softmax_w/Initializer/random_uniform/sub*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
$softmax_w/Initializer/random_uniformAdd(softmax_w/Initializer/random_uniform/mul(softmax_w/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
	softmax_w
VariableV2*
	container *
shape:
��/*
dtype0* 
_output_shapes
:
��/*
shared_name *
_class
loc:@softmax_w
�
softmax_w/AssignAssign	softmax_w$softmax_w/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@softmax_w*
validate_shape(* 
_output_shapes
:
��/
n
softmax_w/readIdentity	softmax_w*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
*softmax_b/Initializer/random_uniform/shapeConst*
_class
loc:@softmax_b*
valueB:�/*
dtype0*
_output_shapes
:
�
(softmax_b/Initializer/random_uniform/minConst*
_class
loc:@softmax_b*
valueB
 *����*
dtype0*
_output_shapes
: 
�
(softmax_b/Initializer/random_uniform/maxConst*
_class
loc:@softmax_b*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
2softmax_b/Initializer/random_uniform/RandomUniformRandomUniform*softmax_b/Initializer/random_uniform/shape*
dtype0*
_output_shapes	
:�/*

seed *
T0*
_class
loc:@softmax_b*
seed2 
�
(softmax_b/Initializer/random_uniform/subSub(softmax_b/Initializer/random_uniform/max(softmax_b/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@softmax_b
�
(softmax_b/Initializer/random_uniform/mulMul2softmax_b/Initializer/random_uniform/RandomUniform(softmax_b/Initializer/random_uniform/sub*
T0*
_class
loc:@softmax_b*
_output_shapes	
:�/
�
$softmax_b/Initializer/random_uniformAdd(softmax_b/Initializer/random_uniform/mul(softmax_b/Initializer/random_uniform/min*
_output_shapes	
:�/*
T0*
_class
loc:@softmax_b
�
	softmax_b
VariableV2*
dtype0*
_output_shapes	
:�/*
shared_name *
_class
loc:@softmax_b*
	container *
shape:�/
�
softmax_b/AssignAssign	softmax_b$softmax_b/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@softmax_b*
validate_shape(*
_output_shapes	
:�/
i
softmax_b/readIdentity	softmax_b*
T0*
_class
loc:@softmax_b*
_output_shapes	
:�/
�
softmax/MatMulMatMul
lm/Reshapesoftmax_w/read*
T0*(
_output_shapes
:����������/*
transpose_a( *
transpose_b( 
�
softmax/BiasAddBiasAddsoftmax/MatMulsoftmax_b/read*
T0*
data_formatNHWC*(
_output_shapes
:����������/
`
Reshape/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
r
ReshapeReshapeinput/Placeholder_1Reshape/shape*
T0*
Tshape0*#
_output_shapes
:���������
U
one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
P
one_hot/depthConst*
value
B :�/*
dtype0*
_output_shapes
: 
�
one_hotOneHotReshapeone_hot/depthone_hot/on_valueone_hot/off_value*(
_output_shapes
:����������/*
T0*
axis���������*
TI0
�
>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientone_hot*(
_output_shapes
:����������/*
T0
p
.loss/softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 
~
/loss/softmax_cross_entropy_with_logits_sg/ShapeShapesoftmax/BiasAdd*
T0*
out_type0*
_output_shapes
:
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
�
1loss/softmax_cross_entropy_with_logits_sg/Shape_1Shapesoftmax/BiasAdd*
T0*
out_type0*
_output_shapes
:
q
/loss/softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
-loss/softmax_cross_entropy_with_logits_sg/SubSub0loss/softmax_cross_entropy_with_logits_sg/Rank_1/loss/softmax_cross_entropy_with_logits_sg/Sub/y*
_output_shapes
: *
T0
�
5loss/softmax_cross_entropy_with_logits_sg/Slice/beginPack-loss/softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
~
4loss/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
/loss/softmax_cross_entropy_with_logits_sg/SliceSlice1loss/softmax_cross_entropy_with_logits_sg/Shape_15loss/softmax_cross_entropy_with_logits_sg/Slice/begin4loss/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:
�
9loss/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
w
5loss/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
0loss/softmax_cross_entropy_with_logits_sg/concatConcatV29loss/softmax_cross_entropy_with_logits_sg/concat/values_0/loss/softmax_cross_entropy_with_logits_sg/Slice5loss/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
1loss/softmax_cross_entropy_with_logits_sg/ReshapeReshapesoftmax/BiasAdd0loss/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:������������������
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
�
1loss/softmax_cross_entropy_with_logits_sg/Shape_2Shape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
_output_shapes
:*
T0*
out_type0
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
/loss/softmax_cross_entropy_with_logits_sg/Sub_1Sub0loss/softmax_cross_entropy_with_logits_sg/Rank_21loss/softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
�
7loss/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack/loss/softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N*
_output_shapes
:
�
6loss/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
1loss/softmax_cross_entropy_with_logits_sg/Slice_1Slice1loss/softmax_cross_entropy_with_logits_sg/Shape_27loss/softmax_cross_entropy_with_logits_sg/Slice_1/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
y
7loss/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
2loss/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_01loss/softmax_cross_entropy_with_logits_sg/Slice_17loss/softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
3loss/softmax_cross_entropy_with_logits_sg/Reshape_1Reshape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient2loss/softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
)loss/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits1loss/softmax_cross_entropy_with_logits_sg/Reshape3loss/softmax_cross_entropy_with_logits_sg/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
�
/loss/softmax_cross_entropy_with_logits_sg/Sub_2Sub.loss/softmax_cross_entropy_with_logits_sg/Rank1loss/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
�
7loss/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
_output_shapes
:*
valueB: *
dtype0
�
6loss/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack/loss/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
1loss/softmax_cross_entropy_with_logits_sg/Slice_2Slice/loss/softmax_cross_entropy_with_logits_sg/Shape7loss/softmax_cross_entropy_with_logits_sg/Slice_2/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
�
3loss/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape)loss/softmax_cross_entropy_with_logits_sg1loss/softmax_cross_entropy_with_logits_sg/Slice_2*#
_output_shapes
:���������*
T0*
Tshape0
T

loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
	loss/MeanMean3loss/softmax_cross_entropy_with_logits_sg/Reshape_2
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
loss_1/tagsConst*
valueB Bloss_1*
dtype0*
_output_shapes
: 
P
loss_1ScalarSummaryloss_1/tags	loss/Mean*
T0*
_output_shapes
: 
�
train/gradients/ShapeShape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*#
_output_shapes
:���������*
T0*

index_type0
Y
train/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
train/gradients/f_count_1Entertrain/gradients/f_count*
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
�
train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
q
train/gradients/SwitchSwitchtrain/gradients/Mergelm/rnn/while/LoopCond*
_output_shapes
: : *
T0
o
train/gradients/Add/yConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
T0*
_output_shapes
: 
�
train/gradients/NextIterationNextIterationtrain/gradients/Addd^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2j^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPushV2h^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPushV2`^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
Z
train/gradients/f_count_2Exittrain/gradients/Switch*
T0*
_output_shapes
: 
Y
train/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
�
train/gradients/Merge_1Mergetrain/gradients/b_count_1train/gradients/NextIteration_1*
_output_shapes
: : *
T0*
N
�
train/gradients/GreaterEqualGreaterEqualtrain/gradients/Merge_1"train/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
�
"train/gradients/GreaterEqual/EnterEntertrain/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
[
train/gradients/b_count_2LoopCondtrain/gradients/GreaterEqual*
_output_shapes
: 
y
train/gradients/Switch_1Switchtrain/gradients/Merge_1train/gradients/b_count_2*
_output_shapes
: : *
T0
{
train/gradients/SubSubtrain/gradients/Switch_1:1"train/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
�
train/gradients/NextIteration_1NextIterationtrain/gradients/Sub_^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
\
train/gradients/b_count_3Exittrain/gradients/Switch_1*
T0*
_output_shapes
: 
�
Ntrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape)loss/softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
�
Ptrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapetrain/gradients/FillNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
train/gradients/zeros_like	ZerosLike+loss/softmax_cross_entropy_with_logits_sg:1*0
_output_shapes
:������������������*
T0
�
Mtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Itrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsPtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeMtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulMulItrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:������������������
�
Itrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax1loss/softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:������������������
�
Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/NegNegItrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:������������������
�
Otrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Ktrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsPtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeOtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
Dtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1MulKtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:������������������
�
Ltrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapesoftmax/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Ntrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeBtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulLtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������/
�
0train/gradients/softmax/BiasAdd_grad/BiasAddGradBiasAddGradNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes	
:�/
�
*train/gradients/softmax/MatMul_grad/MatMulMatMulNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshapesoftmax_w/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/softmax/MatMul_grad/MatMul_1MatMul
lm/ReshapeNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape* 
_output_shapes
:
��/*
transpose_a(*
transpose_b( *
T0
w
%train/gradients/lm/Reshape_grad/ShapeShapelm/rnn/transpose_1*
_output_shapes
:*
T0*
out_type0
�
'train/gradients/lm/Reshape_grad/ReshapeReshape*train/gradients/softmax/MatMul_grad/MatMul%train/gradients/lm/Reshape_grad/Shape*
T0*
Tshape0*,
_output_shapes
:@����������
�
9train/gradients/lm/rnn/transpose_1_grad/InvertPermutationInvertPermutationlm/rnn/concat_2*
T0*
_output_shapes
:
�
1train/gradients/lm/rnn/transpose_1_grad/transpose	Transpose'train/gradients/lm/Reshape_grad/Reshape9train/gradients/lm/rnn/transpose_1_grad/InvertPermutation*
T0*,
_output_shapes
:���������@�*
Tperm0
�
btrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lm/rnn/TensorArraylm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
sourcetrain/gradients*
_output_shapes

:: 
�
^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylm/rnn/while/Exit_2c^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*%
_class
loc:@lm/rnn/TensorArray*
_output_shapes
: 
�
htrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3btrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3lm/rnn/TensorArrayStack/range1train/gradients/lm/rnn/transpose_1_grad/transpose^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
v
%train/gradients/zeros/shape_as_tensorConst*
valueB"@   �   *
dtype0*
_output_shapes
:
`
train/gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zerosFill%train/gradients/zeros/shape_as_tensortrain/gradients/zeros/Const*
T0*

index_type0*
_output_shapes
:	@�
x
'train/gradients/zeros_1/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"@   �   
b
train/gradients/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_1Fill'train/gradients/zeros_1/shape_as_tensortrain/gradients/zeros_1/Const*
_output_shapes
:	@�*
T0*

index_type0
x
'train/gradients/zeros_2/shape_as_tensorConst*
valueB"@   �   *
dtype0*
_output_shapes
:
b
train/gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_2Fill'train/gradients/zeros_2/shape_as_tensortrain/gradients/zeros_2/Const*
T0*

index_type0*
_output_shapes
:	@�
x
'train/gradients/zeros_3/shape_as_tensorConst*
valueB"@   �   *
dtype0*
_output_shapes
:
b
train/gradients/zeros_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
train/gradients/zeros_3Fill'train/gradients/zeros_3/shape_as_tensortrain/gradients/zeros_3/Const*
T0*

index_type0*
_output_shapes
:	@�
�
/train/gradients/lm/rnn/while/Exit_2_grad/b_exitEnterhtrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
�
/train/gradients/lm/rnn/while/Exit_3_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
/train/gradients/lm/rnn/while/Exit_4_grad/b_exitEntertrain/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
/train/gradients/lm/rnn/while/Exit_5_grad/b_exitEntertrain/gradients/zeros_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@�*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
/train/gradients/lm/rnn/while/Exit_6_grad/b_exitEntertrain/gradients/zeros_3*
is_constant( *
parallel_iterations *
_output_shapes
:	@�*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
3train/gradients/lm/rnn/while/Switch_2_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_2_grad/b_exit:train/gradients/lm/rnn/while/Switch_2_grad_1/NextIteration*
_output_shapes
: : *
T0*
N
�
3train/gradients/lm/rnn/while/Switch_3_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_3_grad/b_exit:train/gradients/lm/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@�: 
�
3train/gradients/lm/rnn/while/Switch_4_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_4_grad/b_exit:train/gradients/lm/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@�: 
�
3train/gradients/lm/rnn/while/Switch_5_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_5_grad/b_exit:train/gradients/lm/rnn/while/Switch_5_grad_1/NextIteration*
N*!
_output_shapes
:	@�: *
T0
�
3train/gradients/lm/rnn/while/Switch_6_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_6_grad/b_exit:train/gradients/lm/rnn/while/Switch_6_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@�: 
�
0train/gradients/lm/rnn/while/Merge_2_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 
�
0train/gradients/lm/rnn/while/Merge_3_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_3_grad/b_switch**
_output_shapes
:	@�:	@�*
T0
�
0train/gradients/lm/rnn/while/Merge_4_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_4_grad/b_switchtrain/gradients/b_count_2**
_output_shapes
:	@�:	@�*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_4_grad/b_switch
�
0train/gradients/lm/rnn/while/Merge_5_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_5_grad/b_switchtrain/gradients/b_count_2**
_output_shapes
:	@�:	@�*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_5_grad/b_switch
�
0train/gradients/lm/rnn/while/Merge_6_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_6_grad/b_switchtrain/gradients/b_count_2*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_6_grad/b_switch**
_output_shapes
:	@�:	@�
�
.train/gradients/lm/rnn/while/Enter_2_grad/ExitExit0train/gradients/lm/rnn/while/Merge_2_grad/Switch*
T0*
_output_shapes
: 
�
.train/gradients/lm/rnn/while/Enter_3_grad/ExitExit0train/gradients/lm/rnn/while/Merge_3_grad/Switch*
T0*
_output_shapes
:	@�
�
.train/gradients/lm/rnn/while/Enter_4_grad/ExitExit0train/gradients/lm/rnn/while/Merge_4_grad/Switch*
T0*
_output_shapes
:	@�
�
.train/gradients/lm/rnn/while/Enter_5_grad/ExitExit0train/gradients/lm/rnn/while/Merge_5_grad/Switch*
_output_shapes
:	@�*
T0
�
.train/gradients/lm/rnn/while/Enter_6_grad/ExitExit0train/gradients/lm/rnn/while/Merge_6_grad/Switch*
_output_shapes
:	@�*
T0
�
gtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3mtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
sourcetrain/gradients*
_output_shapes

:: 
�
mtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlm/rnn/TensorArray*
is_constant(*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
parallel_iterations 
�
ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1h^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
_output_shapes
: 
�
Wtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3gtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:	@�
�
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst**
_class 
loc:@lm/rnn/while/Identity_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
_output_shapes
:*
	elem_type0**
_class 
loc:@lm/rnn/while/Identity_1
�
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlm/rnn/while/Identity_1^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
�
htrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerc^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2i^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2g^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2_^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
�
train/gradients/AddNAddN2train/gradients/lm/rnn/while/Merge_6_grad/Switch:1Wtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
N*
_output_shapes
:	@�*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_6_grad/b_switch
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/MulMultrain/gradients/AddN`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1Multrain/gradients/AddNbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@�
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3*

stack_name *
_output_shapes
:*
	elem_type0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3_grad/TanhGradTanhGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul*
_output_shapes
:	@�*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1*
T0*
_output_shapes
:	@�
�
:train/gradients/lm/rnn/while/Switch_2_grad_1/NextIterationNextIteration2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1*
T0*
_output_shapes
: 
�
train/gradients/AddN_1AddN2train/gradients/lm/rnn/while/Merge_5_grad/Switch:1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3_grad/TanhGrad*
N*
_output_shapes
:	@�*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_5_grad/b_switch
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/MulMultrain/gradients/AddN_1`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*

stack_name *
_output_shapes
:*
	elem_type0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1Multrain/gradients/AddN_1btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2*
_output_shapes
:	@�*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/ConstConst**
_class 
loc:@lm/rnn/while/Identity_5*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Const**
_class 
loc:@lm/rnn/while/Identity_5*

stack_name *
_output_shapes
:*
	elem_type0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Enterlm/rnn/while/Identity_5^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/MulMultrain/gradients/AddN_1`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*

stack_name *
_output_shapes
:*
	elem_type0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1Multrain/gradients/AddN_1btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2*
_output_shapes
:	@�*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4*

stack_name *
_output_shapes
:*
	elem_type0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1*
T0*
_output_shapes
:	@�
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4_grad/SigmoidGradSigmoidGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2_grad/TanhGradTanhGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1*
_output_shapes
:	@�*
T0
�
:train/gradients/lm/rnn/while/Switch_5_grad_1/NextIterationNextIterationUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul*
T0*
_output_shapes
:	@�
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ShapeConst^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ShapeYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/SumSumatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradgtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ReshapeReshapeUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/SumWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape*
_output_shapes
:	@�*
T0*
Tshape0
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Sum_1Sumatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGraditrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Reshape_1ReshapeWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Sum_1Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concatConcatV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4_grad/SigmoidGrad[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2_grad/TanhGradYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Reshapeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5_grad/SigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat/Const*
_output_shapes
:	@�*

Tidx0*
T0*
N
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat/ConstConst^train/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGradBiasAddGradZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulMatMulZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concatatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/Enter*
_output_shapes
:	@�*
transpose_a( *
transpose_b(*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1MatMulhtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/ConstConst*R
_classH
FDloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_accStackV2ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Const*R
_classH
FDloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1*

stack_name *
_output_shapes
:*
	elem_type0
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/EnterEnterctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
itrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPushV2StackPushV2ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Enter?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2
StackPopV2ntrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ntrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2/EnterEnterctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConstConst^train/gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/RankConst^train/gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/modFloorModZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConstYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ShapeConst^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"@   �   
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffsetConcatOffsetXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/modZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1*
N* 
_output_shapes
::
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/SliceSlice[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffsetZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape*
Index0*
T0*
_output_shapes
:	@�
�
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice_1Slice[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffset:1\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1*
Index0*
T0*
_output_shapes
:	@�
�
train/gradients/AddN_2AddN2train/gradients/lm/rnn/while/Merge_4_grad/Switch:1Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_4_grad/b_switch*
N*
_output_shapes
:	@�
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/MulMultrain/gradients/AddN_2`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
_output_shapes
:	@�*
T0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1Multrain/gradients/AddN_2btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
_output_shapes
:	@�*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul*
_output_shapes
:	@�*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1*
T0*
_output_shapes
:	@�
�
:train/gradients/lm/rnn/while/Switch_6_grad_1/NextIterationNextIteration\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice_1*
_output_shapes
:	@�*
T0
�
train/gradients/AddN_3AddN2train/gradients/lm/rnn/while/Merge_3_grad/Switch:1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes
:	@�
�
Strain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/MulMultrain/gradients/AddN_3^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@�
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*
valueB :
���������
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Const*Q
_classG
ECloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/EnterEnterYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Enter>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2dtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
dtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1Multrain/gradients/AddN_3`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@�
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/ConstConst**
_class 
loc:@lm/rnn/while/Identity_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Const**
_class 
loc:@lm/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Enterlm/rnn/while/Identity_3^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/MulMultrain/gradients/AddN_3`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
_output_shapes
:	@�*
T0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*N
_classD
B@loc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0*
_output_shapes
: 
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Const*N
_classD
B@loc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Enter;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh^train/gradients/Add*
_output_shapes
:	@�*
swap_memory( *
T0
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1Multrain/gradients/AddN_3btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@�
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1*
T0*
_output_shapes
:	@�
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul*
T0*
_output_shapes
:	@�
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1*
_output_shapes
:	@�*
T0
�
:train/gradients/lm/rnn/while/Switch_3_grad_1/NextIterationNextIterationStrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul*
T0*
_output_shapes
:	@�
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ShapeConst^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
�
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ShapeWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Strain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/SumSum_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradetrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ReshapeReshapeStrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/SumUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape*
_output_shapes
:	@�*
T0*
Tshape0
�
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Sum_1Sum_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradgtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Reshape_1ReshapeUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Sum_1Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concatConcatV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_grad/TanhGradWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Reshapeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes
:	@�*

Tidx0
�
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMulXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concatatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/Enter*
T0*
_output_shapes
:	@�*
transpose_a( *
transpose_b(
�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMulftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat*
valueB :
���������*
dtype0*
_output_shapes
: 
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnteratrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
�
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat^train/gradients/Add*
T0*
_output_shapes
:	@�*
swap_memory( 
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2ltrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@�*
	elem_type0
�
ltrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnteratrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
train/gradients/AddN_4AddNatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:�*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Mergeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:�: 
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*"
_output_shapes
:�:�*
T0
�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAddbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1train/gradients/AddN_4*
_output_shapes	
:�*
T0
�
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:�*
T0
�
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:�*
T0
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Vtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/modFloorModXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConstWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ShapeConst^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1Const^train/gradients/Sub*
valueB"@   �   *
dtype0*
_output_shapes
:
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetVtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/modXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ShapeZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::
�
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/SliceSliceYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape*
_output_shapes
:	@�*
Index0*
T0
�
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Slice_1SliceYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1*
Index0*
T0*
_output_shapes
:	@�
�
train/gradients/AddN_5AddN]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1*
T0*p
_classf
dbloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1*
N* 
_output_shapes
:
��
�
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
��*    *
dtype0* 
_output_shapes
:
��
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
��*:

frame_name,*train/gradients/lm/rnn/while/while_context
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
��: 
�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*,
_output_shapes
:
��:
��*
T0
�
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAddatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1train/gradients/AddN_5*
T0* 
_output_shapes
:
��
�
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIteration\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
��*
T0
�
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
��
�
Utrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3[train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^train/gradients/Sub*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
sourcetrain/gradients*
_output_shapes

:: 
�
[train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlm/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
Qtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1V^train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
�
Wtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Utrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/SliceQtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
�
Atrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterAtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
�
Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Itrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Btrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2train/gradients/b_count_2*
_output_shapes
: : *
T0
�
?train/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddDtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Wtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
Itrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration?train/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0
�
Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitBtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
�
:train/gradients/lm/rnn/while/Switch_4_grad_1/NextIterationNextIterationZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Slice_1*
T0*
_output_shapes
:	@�
�
xtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lm/rnn/TensorArray_1Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *'
_class
loc:@lm/rnn/TensorArray_1*
sourcetrain/gradients
�
ttrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3y^train/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*'
_class
loc:@lm/rnn/TensorArray_1*
_output_shapes
: 
�
jtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3xtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lm/rnn/TensorArrayUnstack/rangettrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*,
_output_shapes
:���������@�
�
7train/gradients/lm/rnn/transpose_grad/InvertPermutationInvertPermutationlm/rnn/concat*
T0*
_output_shapes
:
�
/train/gradients/lm/rnn/transpose_grad/transpose	Transposejtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV37train/gradients/lm/rnn/transpose_grad/InvertPermutation*
Tperm0*
T0*,
_output_shapes
:@����������
�
.train/gradients/lm/embedding_lookup_grad/ShapeConst*
_class
loc:@embedding*%
valueB	"�      �       *
dtype0	*
_output_shapes
:
�
0train/gradients/lm/embedding_lookup_grad/ToInt32Cast.train/gradients/lm/embedding_lookup_grad/Shape*

SrcT0	*
_class
loc:@embedding*
_output_shapes
:*

DstT0
y
-train/gradients/lm/embedding_lookup_grad/SizeSizeinput/Placeholder*
T0*
out_type0*
_output_shapes
: 
y
7train/gradients/lm/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
3train/gradients/lm/embedding_lookup_grad/ExpandDims
ExpandDims-train/gradients/lm/embedding_lookup_grad/Size7train/gradients/lm/embedding_lookup_grad/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
�
<train/gradients/lm/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
�
>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/lm/embedding_lookup_grad/strided_sliceStridedSlice0train/gradients/lm/embedding_lookup_grad/ToInt32<train/gradients/lm/embedding_lookup_grad/strided_slice/stack>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_1>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_2*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask
v
4train/gradients/lm/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
/train/gradients/lm/embedding_lookup_grad/concatConcatV23train/gradients/lm/embedding_lookup_grad/ExpandDims6train/gradients/lm/embedding_lookup_grad/strided_slice4train/gradients/lm/embedding_lookup_grad/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
0train/gradients/lm/embedding_lookup_grad/ReshapeReshape/train/gradients/lm/rnn/transpose_grad/transpose/train/gradients/lm/embedding_lookup_grad/concat*(
_output_shapes
:����������*
T0*
Tshape0
�
2train/gradients/lm/embedding_lookup_grad/Reshape_1Reshapeinput/Placeholder3train/gradients/lm/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:���������
�
train/global_norm/L2LossL2Loss0train/gradients/lm/embedding_lookup_grad/Reshape*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape*
_output_shapes
: 
�
train/global_norm/L2Loss_1L2Loss`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3
�
train/global_norm/L2Loss_2L2Lossatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: 
�
train/global_norm/L2Loss_3L2Loss,train/gradients/softmax/MatMul_grad/MatMul_1*
_output_shapes
: *
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1
�
train/global_norm/L2Loss_4L2Loss0train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
train/global_norm/stackPacktrain/global_norm/L2Losstrain/global_norm/L2Loss_1train/global_norm/L2Loss_2train/global_norm/L2Loss_3train/global_norm/L2Loss_4*
N*
_output_shapes
:*
T0*

axis 
a
train/global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
train/global_norm/SumSumtrain/global_norm/stacktrain/global_norm/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
^
train/global_norm/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
o
train/global_norm/mulMultrain/global_norm/Sumtrain/global_norm/Const_1*
T0*
_output_shapes
: 
]
train/global_norm/global_normSqrttrain/global_norm/mul*
T0*
_output_shapes
: 
h
#train/clip_by_global_norm/truediv/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
!train/clip_by_global_norm/truedivRealDiv#train/clip_by_global_norm/truediv/xtrain/global_norm/global_norm*
T0*
_output_shapes
: 
d
train/clip_by_global_norm/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
j
%train/clip_by_global_norm/truediv_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �@
�
#train/clip_by_global_norm/truediv_1RealDivtrain/clip_by_global_norm/Const%train/clip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 
�
!train/clip_by_global_norm/MinimumMinimum!train/clip_by_global_norm/truediv#train/clip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
d
train/clip_by_global_norm/mul/xConst*
valueB
 *  �@*
dtype0*
_output_shapes
: 
�
train/clip_by_global_norm/mulMultrain/clip_by_global_norm/mul/x!train/clip_by_global_norm/Minimum*
T0*
_output_shapes
: 
�
train/clip_by_global_norm/mul_1Mul0train/gradients/lm/embedding_lookup_grad/Reshapetrain/clip_by_global_norm/mul*(
_output_shapes
:����������*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape
�
6train/clip_by_global_norm/train/clip_by_global_norm/_0Identitytrain/clip_by_global_norm/mul_1*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape*(
_output_shapes
:����������*
T0
�
train/clip_by_global_norm/mul_2Mul`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3train/clip_by_global_norm/mul*
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��
�
6train/clip_by_global_norm/train/clip_by_global_norm/_1Identitytrain/clip_by_global_norm/mul_2*
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:
��
�
train/clip_by_global_norm/mul_3Mulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3train/clip_by_global_norm/mul*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:�
�
6train/clip_by_global_norm/train/clip_by_global_norm/_2Identitytrain/clip_by_global_norm/mul_3*
_output_shapes	
:�*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
�
train/clip_by_global_norm/mul_4Mul,train/gradients/softmax/MatMul_grad/MatMul_1train/clip_by_global_norm/mul* 
_output_shapes
:
��/*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1
�
6train/clip_by_global_norm/train/clip_by_global_norm/_3Identitytrain/clip_by_global_norm/mul_4*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1* 
_output_shapes
:
��/*
T0
�
train/clip_by_global_norm/mul_5Mul0train/gradients/softmax/BiasAdd_grad/BiasAddGradtrain/clip_by_global_norm/mul*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�/
�
6train/clip_by_global_norm/train/clip_by_global_norm/_4Identitytrain/clip_by_global_norm/mul_5*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�/
�
train/beta1_power/initial_valueConst*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *fff?*
dtype0
�
train/beta1_power
VariableV2*
_output_shapes
: *
shared_name *
_class
loc:@embedding*
	container *
shape: *
dtype0
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
t
train/beta1_power/readIdentitytrain/beta1_power*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *w�?
�
train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@embedding
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(
t
train/beta2_power/readIdentitytrain/beta2_power*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
0embedding/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@embedding*
valueB"�  �   *
dtype0*
_output_shapes
:
�
&embedding/Adam/Initializer/zeros/ConstConst*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 embedding/Adam/Initializer/zerosFill0embedding/Adam/Initializer/zeros/shape_as_tensor&embedding/Adam/Initializer/zeros/Const*
T0*
_class
loc:@embedding*

index_type0* 
_output_shapes
:
�/�
�
embedding/Adam
VariableV2*
shared_name *
_class
loc:@embedding*
	container *
shape:
�/�*
dtype0* 
_output_shapes
:
�/�
�
embedding/Adam/AssignAssignembedding/Adam embedding/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�
x
embedding/Adam/readIdentityembedding/Adam* 
_output_shapes
:
�/�*
T0*
_class
loc:@embedding
�
2embedding/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@embedding*
valueB"�  �   *
dtype0*
_output_shapes
:
�
(embedding/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"embedding/Adam_1/Initializer/zerosFill2embedding/Adam_1/Initializer/zeros/shape_as_tensor(embedding/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@embedding*

index_type0* 
_output_shapes
:
�/�
�
embedding/Adam_1
VariableV2*
shape:
�/�*
dtype0* 
_output_shapes
:
�/�*
shared_name *
_class
loc:@embedding*
	container 
�
embedding/Adam_1/AssignAssignembedding/Adam_1"embedding/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�
|
embedding/Adam_1/readIdentityembedding/Adam_1*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
Wrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zerosFillWrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorMrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*

index_type0* 
_output_shapes
:
��
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
��
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��
�
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *    
�
Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*

index_type0* 
_output_shapes
:
��
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��
�
Ernn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:�
�
8rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:�
�
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container 
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:�*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
�
0softmax_w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_w*
valueB"�   �  *
dtype0*
_output_shapes
:
�
&softmax_w/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@softmax_w*
valueB
 *    
�
 softmax_w/Adam/Initializer/zerosFill0softmax_w/Adam/Initializer/zeros/shape_as_tensor&softmax_w/Adam/Initializer/zeros/Const*
_class
loc:@softmax_w*

index_type0* 
_output_shapes
:
��/*
T0
�
softmax_w/Adam
VariableV2*
	container *
shape:
��/*
dtype0* 
_output_shapes
:
��/*
shared_name *
_class
loc:@softmax_w
�
softmax_w/Adam/AssignAssignsoftmax_w/Adam softmax_w/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��/*
use_locking(*
T0*
_class
loc:@softmax_w
x
softmax_w/Adam/readIdentitysoftmax_w/Adam*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
2softmax_w/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@softmax_w*
valueB"�   �  
�
(softmax_w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@softmax_w*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_w/Adam_1/Initializer/zerosFill2softmax_w/Adam_1/Initializer/zeros/shape_as_tensor(softmax_w/Adam_1/Initializer/zeros/Const*
_class
loc:@softmax_w*

index_type0* 
_output_shapes
:
��/*
T0
�
softmax_w/Adam_1
VariableV2*
shape:
��/*
dtype0* 
_output_shapes
:
��/*
shared_name *
_class
loc:@softmax_w*
	container 
�
softmax_w/Adam_1/AssignAssignsoftmax_w/Adam_1"softmax_w/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��/*
use_locking(*
T0*
_class
loc:@softmax_w
|
softmax_w/Adam_1/readIdentitysoftmax_w/Adam_1*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
��/
�
0softmax_b/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_b*
valueB:�/*
dtype0*
_output_shapes
:
�
&softmax_b/Adam/Initializer/zeros/ConstConst*
_class
loc:@softmax_b*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 softmax_b/Adam/Initializer/zerosFill0softmax_b/Adam/Initializer/zeros/shape_as_tensor&softmax_b/Adam/Initializer/zeros/Const*
_output_shapes	
:�/*
T0*
_class
loc:@softmax_b*

index_type0
�
softmax_b/Adam
VariableV2*
dtype0*
_output_shapes	
:�/*
shared_name *
_class
loc:@softmax_b*
	container *
shape:�/
�
softmax_b/Adam/AssignAssignsoftmax_b/Adam softmax_b/Adam/Initializer/zeros*
_class
loc:@softmax_b*
validate_shape(*
_output_shapes	
:�/*
use_locking(*
T0
s
softmax_b/Adam/readIdentitysoftmax_b/Adam*
T0*
_class
loc:@softmax_b*
_output_shapes	
:�/
�
2softmax_b/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_b*
valueB:�/*
dtype0*
_output_shapes
:
�
(softmax_b/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@softmax_b*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_b/Adam_1/Initializer/zerosFill2softmax_b/Adam_1/Initializer/zeros/shape_as_tensor(softmax_b/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�/*
T0*
_class
loc:@softmax_b*

index_type0
�
softmax_b/Adam_1
VariableV2*
shared_name *
_class
loc:@softmax_b*
	container *
shape:�/*
dtype0*
_output_shapes	
:�/
�
softmax_b/Adam_1/AssignAssignsoftmax_b/Adam_1"softmax_b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@softmax_b*
validate_shape(*
_output_shapes	
:�/
w
softmax_b/Adam_1/readIdentitysoftmax_b/Adam_1*
_class
loc:@softmax_b*
_output_shapes	
:�/*
T0
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
"train/Adam/update_embedding/UniqueUnique2train/gradients/lm/embedding_lookup_grad/Reshape_1*2
_output_shapes 
:���������:���������*
out_idx0*
T0*
_class
loc:@embedding
�
!train/Adam/update_embedding/ShapeShape"train/Adam/update_embedding/Unique*
T0*
_class
loc:@embedding*
out_type0*
_output_shapes
:
�
/train/Adam/update_embedding/strided_slice/stackConst*
_class
loc:@embedding*
valueB: *
dtype0*
_output_shapes
:
�
1train/Adam/update_embedding/strided_slice/stack_1Const*
_class
loc:@embedding*
valueB:*
dtype0*
_output_shapes
:
�
1train/Adam/update_embedding/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
_class
loc:@embedding*
valueB:
�
)train/Adam/update_embedding/strided_sliceStridedSlice!train/Adam/update_embedding/Shape/train/Adam/update_embedding/strided_slice/stack1train/Adam/update_embedding/strided_slice/stack_11train/Adam/update_embedding/strided_slice/stack_2*
Index0*
T0*
_class
loc:@embedding*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
.train/Adam/update_embedding/UnsortedSegmentSumUnsortedSegmentSum6train/clip_by_global_norm/train/clip_by_global_norm/_0$train/Adam/update_embedding/Unique:1)train/Adam/update_embedding/strided_slice*
Tnumsegments0*
Tindices0*
T0*
_class
loc:@embedding*(
_output_shapes
:����������
�
!train/Adam/update_embedding/sub/xConst*
_class
loc:@embedding*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/Adam/update_embedding/subSub!train/Adam/update_embedding/sub/xtrain/beta2_power/read*
_class
loc:@embedding*
_output_shapes
: *
T0
�
 train/Adam/update_embedding/SqrtSqrttrain/Adam/update_embedding/sub*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
train/Adam/update_embedding/mulMullearning_rate/Variable/read train/Adam/update_embedding/Sqrt*
_output_shapes
: *
T0*
_class
loc:@embedding
�
#train/Adam/update_embedding/sub_1/xConst*
_class
loc:@embedding*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!train/Adam/update_embedding/sub_1Sub#train/Adam/update_embedding/sub_1/xtrain/beta1_power/read*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
#train/Adam/update_embedding/truedivRealDivtrain/Adam/update_embedding/mul!train/Adam/update_embedding/sub_1*
_output_shapes
: *
T0*
_class
loc:@embedding
�
#train/Adam/update_embedding/sub_2/xConst*
_class
loc:@embedding*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!train/Adam/update_embedding/sub_2Sub#train/Adam/update_embedding/sub_2/xtrain/Adam/beta1*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
!train/Adam/update_embedding/mul_1Mul.train/Adam/update_embedding/UnsortedSegmentSum!train/Adam/update_embedding/sub_2*
T0*
_class
loc:@embedding*(
_output_shapes
:����������
�
!train/Adam/update_embedding/mul_2Mulembedding/Adam/readtrain/Adam/beta1*
_class
loc:@embedding* 
_output_shapes
:
�/�*
T0
�
"train/Adam/update_embedding/AssignAssignembedding/Adam!train/Adam/update_embedding/mul_2*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�
�
&train/Adam/update_embedding/ScatterAdd
ScatterAddembedding/Adam"train/Adam/update_embedding/Unique!train/Adam/update_embedding/mul_1#^train/Adam/update_embedding/Assign*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
!train/Adam/update_embedding/mul_3Mul.train/Adam/update_embedding/UnsortedSegmentSum.train/Adam/update_embedding/UnsortedSegmentSum*
T0*
_class
loc:@embedding*(
_output_shapes
:����������
�
#train/Adam/update_embedding/sub_3/xConst*
_class
loc:@embedding*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!train/Adam/update_embedding/sub_3Sub#train/Adam/update_embedding/sub_3/xtrain/Adam/beta2*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
!train/Adam/update_embedding/mul_4Mul!train/Adam/update_embedding/mul_3!train/Adam/update_embedding/sub_3*(
_output_shapes
:����������*
T0*
_class
loc:@embedding
�
!train/Adam/update_embedding/mul_5Mulembedding/Adam_1/readtrain/Adam/beta2*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
$train/Adam/update_embedding/Assign_1Assignembedding/Adam_1!train/Adam/update_embedding/mul_5*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
�/�
�
(train/Adam/update_embedding/ScatterAdd_1
ScatterAddembedding/Adam_1"train/Adam/update_embedding/Unique!train/Adam/update_embedding/mul_4%^train/Adam/update_embedding/Assign_1*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
"train/Adam/update_embedding/Sqrt_1Sqrt(train/Adam/update_embedding/ScatterAdd_1*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
!train/Adam/update_embedding/mul_6Mul#train/Adam/update_embedding/truediv&train/Adam/update_embedding/ScatterAdd*
T0*
_class
loc:@embedding* 
_output_shapes
:
�/�
�
train/Adam/update_embedding/addAdd"train/Adam/update_embedding/Sqrt_1train/Adam/epsilon* 
_output_shapes
:
�/�*
T0*
_class
loc:@embedding
�
%train/Adam/update_embedding/truediv_1RealDiv!train/Adam/update_embedding/mul_6train/Adam/update_embedding/add*
_class
loc:@embedding* 
_output_shapes
:
�/�*
T0
�
%train/Adam/update_embedding/AssignSub	AssignSub	embedding%train/Adam/update_embedding/truediv_1* 
_output_shapes
:
�/�*
use_locking( *
T0*
_class
loc:@embedding
�
&train/Adam/update_embedding/group_depsNoOp&^train/Adam/update_embedding/AssignSub'^train/Adam/update_embedding/ScatterAdd)^train/Adam/update_embedding/ScatterAdd_1*
_class
loc:@embedding
�
Ltrain/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_1*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
Jtrain/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_2*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
%train/Adam/update_softmax_w/ApplyAdam	ApplyAdam	softmax_wsoftmax_w/Adamsoftmax_w/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_3*
use_locking( *
T0*
_class
loc:@softmax_w*
use_nesterov( * 
_output_shapes
:
��/
�
%train/Adam/update_softmax_b/ApplyAdam	ApplyAdam	softmax_bsoftmax_b/Adamsoftmax_b/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_4*
use_nesterov( *
_output_shapes	
:�/*
use_locking( *
T0*
_class
loc:@softmax_b
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam*
T0*
_class
loc:@embedding*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam*
_class
loc:@embedding*
_output_shapes
: *
T0
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam
�
initNoOp^embedding/Adam/Assign^embedding/Adam_1/Assign^embedding/Assign^learning_rate/Variable/Assign;^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign6^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign8^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign^softmax_b/Adam/Assign^softmax_b/Adam_1/Assign^softmax_b/Assign^softmax_w/Adam/Assign^softmax_w/Adam_1/Assign^softmax_w/Assign^train/beta1_power/Assign^train/beta2_power/Assign
\
Merge/MergeSummaryMergeSummarylearning_rate_1loss_1*
N*
_output_shapes
: ""�
trainable_variables��
Y
embedding:0embedding/Assignembedding/read:02&embedding/Initializer/random_uniform:0
�
2rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zeros:0
Y
softmax_w:0softmax_w/Assignsoftmax_w/read:02&softmax_w/Initializer/random_uniform:0
Y
softmax_b:0softmax_b/Assignsoftmax_b/read:02&softmax_b/Initializer/random_uniform:0",
	summaries

learning_rate_1:0
loss_1:0"
train_op


train/Adam"�l
while_context�l�l
�l
lm/rnn/while/while_context *lm/rnn/while/LoopCond:02lm/rnn/while/Merge:0:lm/rnn/while/Identity:0Blm/rnn/while/Exit:0Blm/rnn/while/Exit_1:0Blm/rnn/while/Exit_2:0Blm/rnn/while/Exit_3:0Blm/rnn/while/Exit_4:0Blm/rnn/while/Exit_5:0Blm/rnn/while/Exit_6:0Btrain/gradients/f_count_2:0J�h
lm/rnn/Minimum:0
lm/rnn/TensorArray:0
Clm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
lm/rnn/TensorArray_1:0
lm/rnn/strided_slice:0
lm/rnn/while/Enter:0
lm/rnn/while/Enter_1:0
lm/rnn/while/Enter_2:0
lm/rnn/while/Enter_3:0
lm/rnn/while/Enter_4:0
lm/rnn/while/Enter_5:0
lm/rnn/while/Enter_6:0
lm/rnn/while/Exit:0
lm/rnn/while/Exit_1:0
lm/rnn/while/Exit_2:0
lm/rnn/while/Exit_3:0
lm/rnn/while/Exit_4:0
lm/rnn/while/Exit_5:0
lm/rnn/while/Exit_6:0
lm/rnn/while/Identity:0
lm/rnn/while/Identity_1:0
lm/rnn/while/Identity_2:0
lm/rnn/while/Identity_3:0
lm/rnn/while/Identity_4:0
lm/rnn/while/Identity_5:0
lm/rnn/while/Identity_6:0
lm/rnn/while/Less/Enter:0
lm/rnn/while/Less:0
lm/rnn/while/Less_1/Enter:0
lm/rnn/while/Less_1:0
lm/rnn/while/LogicalAnd:0
lm/rnn/while/LoopCond:0
lm/rnn/while/Merge:0
lm/rnn/while/Merge:1
lm/rnn/while/Merge_1:0
lm/rnn/while/Merge_1:1
lm/rnn/while/Merge_2:0
lm/rnn/while/Merge_2:1
lm/rnn/while/Merge_3:0
lm/rnn/while/Merge_3:1
lm/rnn/while/Merge_4:0
lm/rnn/while/Merge_4:1
lm/rnn/while/Merge_5:0
lm/rnn/while/Merge_5:1
lm/rnn/while/Merge_6:0
lm/rnn/while/Merge_6:1
lm/rnn/while/NextIteration:0
lm/rnn/while/NextIteration_1:0
lm/rnn/while/NextIteration_2:0
lm/rnn/while/NextIteration_3:0
lm/rnn/while/NextIteration_4:0
lm/rnn/while/NextIteration_5:0
lm/rnn/while/NextIteration_6:0
lm/rnn/while/Switch:0
lm/rnn/while/Switch:1
lm/rnn/while/Switch_1:0
lm/rnn/while/Switch_1:1
lm/rnn/while/Switch_2:0
lm/rnn/while/Switch_2:1
lm/rnn/while/Switch_3:0
lm/rnn/while/Switch_3:1
lm/rnn/while/Switch_4:0
lm/rnn/while/Switch_4:1
lm/rnn/while/Switch_5:0
lm/rnn/while/Switch_5:1
lm/rnn/while/Switch_6:0
lm/rnn/while/Switch_6:1
&lm/rnn/while/TensorArrayReadV3/Enter:0
(lm/rnn/while/TensorArrayReadV3/Enter_1:0
 lm/rnn/while/TensorArrayReadV3:0
8lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
lm/rnn/while/add/y:0
lm/rnn/while/add:0
lm/rnn/while/add_1/y:0
lm/rnn/while/add_1:0
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3:0
Flm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter:0
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd:0
Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const:0
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_1:0
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2:0
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3:0
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_4:0
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5:0
Elm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter:0
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul:0
Alm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1:0
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5:0
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid:0
Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1:0
Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2:0
Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3:0
Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4:0
Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5:0
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh:0
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1:0
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2:0
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3:0
Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axis:0
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat:0
Flm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axis:0
Alm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:0
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:1
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:2
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:3
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:0
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:1
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:2
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:3
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0
train/gradients/Add/y:0
train/gradients/Add:0
train/gradients/Merge:0
train/gradients/Merge:1
train/gradients/NextIteration:0
train/gradients/Switch:0
train/gradients/Switch:1
train/gradients/f_count:0
train/gradients/f_count_1:0
train/gradients/f_count_2:0
_train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
etrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
_train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Enter:0
ktrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPushV2:0
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc:0
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
itrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Enter:0
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Enter:0
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Enter:0
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPushV2:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Enter:0
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPushV2:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Enter:0
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPushV2:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Enter:0
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPushV2:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Enter:0
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPushV2:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Enter:0
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPushV2:0
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc:0
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Enter:0
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPushV2:0
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Enter:0
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2:0
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0�
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc:0etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Enter:0/
lm/rnn/Minimum:0lm/rnn/while/Less_1/Enter:0�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Enter:0�
_train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0_train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0@
lm/rnn/TensorArray_1:0&lm/rnn/while/TensorArrayReadV3/Enter:0�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Enter:0�
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Enter:0�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Enter:0�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0o
Clm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(lm/rnn/while/TensorArrayReadV3/Enter_1:0�
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0P
lm/rnn/TensorArray:08lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Enter:0�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Enter:0�
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc:0[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Enter:0�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Enter:03
lm/rnn/strided_slice:0lm/rnn/while/Less/Enter:0�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Enter:0�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0Elm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter:0
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0Flm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter:0�
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Enter:0Rlm/rnn/while/Enter:0Rlm/rnn/while/Enter_1:0Rlm/rnn/while/Enter_2:0Rlm/rnn/while/Enter_3:0Rlm/rnn/while/Enter_4:0Rlm/rnn/while/Enter_5:0Rlm/rnn/while/Enter_6:0Rtrain/gradients/f_count_1:0Zlm/rnn/strided_slice:0"�
	variables��
�
learning_rate/Variable:0learning_rate/Variable/Assignlearning_rate/Variable/read:02&learning_rate/Variable/initial_value:0
Y
embedding:0embedding/Assignembedding/read:02&embedding/Initializer/random_uniform:0
�
2rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zeros:0
Y
softmax_w:0softmax_w/Assignsoftmax_w/read:02&softmax_w/Initializer/random_uniform:0
Y
softmax_b:0softmax_b/Assignsoftmax_b/read:02&softmax_b/Initializer/random_uniform:0
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
d
embedding/Adam:0embedding/Adam/Assignembedding/Adam/read:02"embedding/Adam/Initializer/zeros:0
l
embedding/Adam_1:0embedding/Adam_1/Assignembedding/Adam_1/read:02$embedding/Adam_1/Initializer/zeros:0
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam:0<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/read:02Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros:0
�
9rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1:0>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/read:02Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam:0:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/read:02Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros:0
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1:0<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/read:02Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
d
softmax_w/Adam:0softmax_w/Adam/Assignsoftmax_w/Adam/read:02"softmax_w/Adam/Initializer/zeros:0
l
softmax_w/Adam_1:0softmax_w/Adam_1/Assignsoftmax_w/Adam_1/read:02$softmax_w/Adam_1/Initializer/zeros:0
d
softmax_b/Adam:0softmax_b/Adam/Assignsoftmax_b/Adam/read:02"softmax_b/Adam/Initializer/zeros:0
l
softmax_b/Adam_1:0softmax_b/Adam_1/Assignsoftmax_b/Adam_1/read:02$softmax_b/Adam_1/Initializer/zeros:0
d�2       $V�	h�˟���A*'

learning_rate_1o;

loss_1�A���84       ^3\	W;џ���A*'

learning_rate_1o;

loss_1VyA2h4       ^3\	�7֟���A*'

learning_rate_1o;

loss_1e`A���Y4       ^3\	Gzܟ���A*'

learning_rate_1o;

loss_1CA�v��4       ^3\	������A*'

learning_rate_1o;

loss_1WA���4       ^3\	�2����A*'

learning_rate_1o;

loss_1ix
Ax�kL4       ^3\	������A*'

learning_rate_1o;

loss_1�a	ACqTx4       ^3\	�}�����A*'

learning_rate_1o;

loss_1�]A�v�4       ^3\	3����A*'

learning_rate_1o;

loss_1)A��4       ^3\	�����A	*'

learning_rate_1o;

loss_1,j�@/�=*4       ^3\	�B����A
*'

learning_rate_1o;

loss_1�U�@ۓ4L4       ^3\	�<����A*'

learning_rate_1o;

loss_1���@���D4       ^3\	ɫ$����A*'

learning_rate_1o;

loss_1��@�e"4       ^3\	�-����A*'

learning_rate_1o;

loss_1D��@�b<�4       ^3\	O5����A*'

learning_rate_1o;

loss_1t|�@Q���4       ^3\	�2=����A*'

learning_rate_1o;

loss_1�-�@���P4       ^3\	ǣE����A*'

learning_rate_1o;

loss_1�t�@t�4       ^3\	��M����A*'

learning_rate_1o;

loss_1�#�@�NH�4       ^3\	�U����A*'

learning_rate_1o;

loss_1���@��-4       ^3\	��^����A*'

learning_rate_1o;

loss_1QY�@0Ї4       ^3\	Q�f����A*'

learning_rate_1o;

loss_1'�@�mT4       ^3\	��o����A*'

learning_rate_1o;

loss_1�3�@T�g4       ^3\	�w����A*'

learning_rate_1o;

loss_1�9�@�"z�4       ^3\	�8�����A*'

learning_rate_1o;

loss_1}�@���4       ^3\	������A*'

learning_rate_1o;

loss_1���@D	��4       ^3\	 <�����A*'

learning_rate_1o;

loss_1�`�@A�.r4       ^3\	�r�����A*'

learning_rate_1o;

loss_1R��@�&[4       ^3\	5������A*'

learning_rate_1o;

loss_1�d�@��4       ^3\	y멠���A*'

learning_rate_1o;

loss_1���@%�4       ^3\	����A*'

learning_rate_1o;

loss_1���@�;t4       ^3\	V�����A*'

learning_rate_1o;

loss_1��@cѕj4       ^3\	�y ���A*'

learning_rate_1o;

loss_1oW�@��RL4       ^3\	p�ʠ���A *'

learning_rate_1o;

loss_1ҝ�@�$�34       ^3\	)�Ҡ���A!*'

learning_rate_1o;

loss_1k��@2#7�4       ^3\	+�ڠ���A"*'

learning_rate_1o;

loss_1���@��/4       ^3\	�L����A#*'

learning_rate_1o;

loss_1�8�@��_�4       ^3\	�J����A$*'

learning_rate_1o;

loss_1/��@m� �4       ^3\	o�����A%*'

learning_rate_1o;

loss_1@A�@�8�U4       ^3\	�������A&*'

learning_rate_1o;

loss_1p��@�(n�4       ^3\	0�����A'*'

learning_rate_1o;

loss_1���@m��4       ^3\	2����A(*'

learning_rate_1o;

loss_1A��@k��4       ^3\	�s����A)*'

learning_rate_1o;

loss_1�{�@��}4       ^3\	$�����A**'

learning_rate_1o;

loss_1 ��@f��4       ^3\	C�%����A+*'

learning_rate_1o;

loss_11��@eE®4       ^3\	Cc/����A,*'

learning_rate_1o;

loss_1D�@>��24       ^3\	�?:����A-*'

learning_rate_1o;

loss_1x��@/��4       ^3\	��D����A.*'

learning_rate_1o;

loss_1E�@� ��4       ^3\	�RO����A/*'

learning_rate_1o;

loss_1��@ὴ4       ^3\	��Y����A0*'

learning_rate_1o;

loss_1h*�@�0i�4       ^3\	�Ud����A1*'

learning_rate_1o;

loss_1�k�@�I0;4       ^3\	��n����A2*'

learning_rate_1o;

loss_1�V�@�T�4       ^3\	e^y����A3*'

learning_rate_1o;

loss_1���@��r�4       ^3\	�������A4*'

learning_rate_1o;

loss_1<Z�@���4       ^3\	�n�����A5*'

learning_rate_1o;

loss_1���@�l\4       ^3\	lØ����A6*'

learning_rate_1o;

loss_1v�@��4       ^3\	�t�����A7*'

learning_rate_1o;

loss_1z��@����4       ^3\	譡���A8*'

learning_rate_1o;

loss_1�D�@�ܚ$4       ^3\	i�����A9*'

learning_rate_1o;

loss_1��@��>�4       ^3\	q�¡���A:*'

learning_rate_1o;

loss_1	��@��Td4       ^3\	�-͡���A;*'

learning_rate_1o;

loss_1�[�@���4       ^3\	��ס���A<*'

learning_rate_1o;

loss_1��@�~<�4       ^3\	�0����A=*'

learning_rate_1o;

loss_1t��@����4       ^3\	j�����A>*'

learning_rate_1o;

loss_1���@��X�4       ^3\	EO�����A?*'

learning_rate_1o;

loss_1#��@��)4       ^3\	¿����A@*'

learning_rate_1o;

loss_1���@�[|4       ^3\	�;����AA*'

learning_rate_1o;

loss_1���@V��4       ^3\	������AB*'

learning_rate_1o;

loss_1�9�@�?�4       ^3\	S5!����AC*'

learning_rate_1o;

loss_1���@!{	�4       ^3\	N�+����AD*'

learning_rate_1o;

loss_1�@�B[�4       ^3\	%6����AE*'

learning_rate_1o;

loss_1��@3kG�4       ^3\	T�@����AF*'

learning_rate_1o;

loss_1�~�@����4       ^3\	MK����AG*'

learning_rate_1o;

loss_1���@T��4       ^3\	�V����AH*'

learning_rate_1o;

loss_1���@�D@�4       ^3\	ֳ`����AI*'

learning_rate_1o;

loss_1���@TԾt4       ^3\	k%k����AJ*'

learning_rate_1o;

loss_1��@c�b�4       ^3\	<�u����AK*'

learning_rate_1o;

loss_1@��@��j4       ^3\	������AL*'

learning_rate_1o;

loss_1���@��T�4       ^3\	H������AM*'

learning_rate_1o;

loss_1Z��@���4       ^3\	�����AN*'

learning_rate_1o;

loss_1h��@2�(4       ^3\	~�����AO*'

learning_rate_1o;

loss_1��@a5��4       ^3\	)ѩ����AP*'

learning_rate_1o;

loss_1C��@ ,�X4       ^3\	@J�����AQ*'

learning_rate_1o;

loss_1��@�Н�4       ^3\	�ܾ����AR*'

learning_rate_1o;

loss_1���@�շ�4       ^3\	O+ɢ���AS*'

learning_rate_1o;

loss_1���@�0u�4       ^3\	��Ӣ���AT*'

learning_rate_1o;

loss_1�~�@Q/4       ^3\	�.ޢ���AU*'

learning_rate_1o;

loss_15M�@�ei4       ^3\	�����AV*'

learning_rate_1o;

loss_1��@\u�4       ^3\	0�����AW*'

learning_rate_1o;

loss_1F��@�:�4       ^3\	Y�����AX*'

learning_rate_1o;

loss_1Z��@��S�4       ^3\	������AY*'

learning_rate_1o;

loss_12^�@�J034       ^3\	WQ����AZ*'

learning_rate_1o;

loss_1��@��4       ^3\	]�����A[*'

learning_rate_1o;

loss_18�@EV��4       ^3\	(����A\*'

learning_rate_1o;

loss_1��@��@�4       ^3\	`�2����A]*'

learning_rate_1o;

loss_10}�@+*v�4       ^3\	��<����A^*'

learning_rate_1o;

loss_17��@@�f4       ^3\	qlG����A_*'

learning_rate_1o;

loss_1��@֯jU4       ^3\	p�Q����A`*'

learning_rate_1o;

loss_1�@ZdfV4       ^3\	l%\����Aa*'

learning_rate_1o;

loss_1I��@�u\�4       ^3\	Q�f����Ab*'

learning_rate_1o;

loss_1���@!m@]4       ^3\		q����Ac*'

learning_rate_1o;

loss_1)W�@�?Bx4       ^3\	!�{����Ad*'

learning_rate_1o;

loss_1p��@���O4       ^3\	.d�����Ae*'

learning_rate_1o;

loss_17��@�q�4       ^3\	7퐣���Af*'

learning_rate_1o;

loss_1��@F�.�4       ^3\	Vy�����Ag*'

learning_rate_1o;

loss_1##�@٬��4       ^3\	������Ah*'

learning_rate_1o;

loss_1X��@�]�4       ^3\	-������Ai*'

learning_rate_1o;

loss_1���@��� 4       ^3\	j2�����Aj*'

learning_rate_1o;

loss_1���@��(�4       ^3\	��ţ���Ak*'

learning_rate_1o;

loss_1^��@)�?D4       ^3\	�@У���Al*'

learning_rate_1o;

loss_1�'�@��	�4       ^3\	��ڣ���Am*'

learning_rate_1o;

loss_1�{�@��_�4       ^3\	
�����An*'

learning_rate_1o;

loss_1ط�@��"�4       ^3\	ݍ����Ao*'

learning_rate_1o;

loss_1x�@k��K4       ^3\	�������Ap*'

learning_rate_1o;

loss_1s��@>��4       ^3\	bZ����Aq*'

learning_rate_1o;

loss_1�!�@���4       ^3\	�����Ar*'

learning_rate_1o;

loss_1�E�@�X4       ^3\	,����As*'

learning_rate_1o;

loss_1m��@"�K4       ^3\	��#����At*'

learning_rate_1o;

loss_1���@�D4       ^3\	� .����Au*'

learning_rate_1o;

loss_1:��@:�9�4       ^3\	�8����Av*'

learning_rate_1o;

loss_1���@g�4       ^3\	#C����Aw*'

learning_rate_1o;

loss_1J��@I��V4       ^3\	U�M����Ax*'

learning_rate_1o;

loss_1.��@X�}Y4       ^3\	Y#X����Ay*'

learning_rate_1o;

loss_17$�@G�4       ^3\	3�b����Az*'

learning_rate_1o;

loss_1@'�@L��4       ^3\	M�m����A{*'

learning_rate_1o;

loss_1�i�@��`�4       ^3\	x����A|*'

learning_rate_1o;

loss_1]��@o�@4       ^3\	������A}*'

learning_rate_1o;

loss_1���@k
�4       ^3\	w@�����A~*'

learning_rate_1o;

loss_1&t�@�-��4       ^3\	�������A*'

learning_rate_1o;

loss_1�D�@kSk5       ��]�	�ݡ����A�*'

learning_rate_1o;

loss_1w�@* ��5       ��]�	T2�����A�*'

learning_rate_1o;

loss_1�I�@����5       ��]�	Jƶ����A�*'

learning_rate_1o;

loss_1�c�@ s��5       ��]�	�E�����A�*'

learning_rate_1o;

loss_1)�@��J�5       ��]�	V�ˤ���A�*'

learning_rate_1o;

loss_1H��@���z5       ��]�	�[֤���A�*'

learning_rate_1o;

loss_1j��@��N�5       ��]�	������A�*'

learning_rate_1o;

loss_1<��@Y�Q5       ��]�	P>����A�*'

learning_rate_1o;

loss_1�L�@!�5       ��]�	�������A�*'

learning_rate_1o;

loss_1�@�c�5       ��]�	m9 ����A�*'

learning_rate_1o;

loss_1�z�@v�(�5       ��]�	�
����A�*'

learning_rate_1o;

loss_1���@$��5       ��]�	2����A�*'

learning_rate_1o;

loss_1l�@��E5       ��]�	������A�*'

learning_rate_1o;

loss_1���@�c�85       ��]�	��)����A�*'

learning_rate_1o;

loss_1�K�@�E<5       ��]�	k4����A�*'

learning_rate_1o;

loss_1���@�I4W5       ��]�	��>����A�*'

learning_rate_1o;

loss_1&'�@�5       ��]�	�EI����A�*'

learning_rate_1o;

loss_1k4�@�zg;5       ��]�	֭S����A�*'

learning_rate_1o;

loss_1���@=sL|5       ��]�	�!^����A�*'

learning_rate_1o;

loss_1�]�@���5       ��]�	
~h����A�*'

learning_rate_1o;

loss_1܄�@�n�5       ��]�	��r����A�*'

learning_rate_1o;

loss_1�W�@��5�5       ��]�	�6}����A�*'

learning_rate_1o;

loss_1���@���B5       ��]�	�������A�*'

learning_rate_1o;

loss_1Q-�@��('5       ��]�	�Ց����A�*'

learning_rate_1o;

loss_1z��@J���5       ��]�	l.�����A�*'

learning_rate_1o;

loss_1���@���5       ��]�	&������A�*'

learning_rate_1o;

loss_1�N�@\y�n5       ��]�	ְ����A�*'

learning_rate_1o;

loss_1���@�=Gt5       ��]�	k�����A�*'

learning_rate_1o;

loss_1��@9�M�5       ��]�	��ť���A�*'

learning_rate_1o;

loss_1��@;��5       ��]�	�Х���A�*'

learning_rate_1o;

loss_1>��@7���5       ��]�	��ڥ���A�*'

learning_rate_1o;

loss_1I��@��8d5       ��]�	������A�*'

learning_rate_1o;

loss_1ϭ�@B�5       ��]�	������A�*'

learning_rate_1o;

loss_1���@�f��5       ��]�	l������A�*'

learning_rate_1o;

loss_1TP�@vH@(5       ��]�	ٚ����A�*'

learning_rate_1o;

loss_1n=�@��v5       ��]�	�Y����A�*'

learning_rate_1o;

loss_1���@��r5       ��]�	�	 ����A�*'

learning_rate_1o;

loss_1H��@����5       ��]�	6w+����A�*'

learning_rate_1o;

loss_1M��@�My5       ��]�	�8����A�*'

learning_rate_1o;

loss_1�v�@��ܱ5       ��]�	��E����A�*'

learning_rate_1o;

loss_1q?�@I���5       ��]�	��R����A�*'

learning_rate_1o;

loss_1=��@TL�5       ��]�	�X_����A�*'

learning_rate_1o;

loss_1��@�Q��5       ��]�	gVl����A�*'

learning_rate_1o;

loss_1 �@$N\e5       ��]�	(�x����A�*'

learning_rate_1o;

loss_1N6�@4�A�5       ��]�	�օ����A�*'

learning_rate_1o;

loss_1��@�ܬ�5       ��]�	�䒦���A�*'

learning_rate_1o;

loss_1?��@�uT5       ��]�		������A�*'

learning_rate_1o;

loss_1�Y�@ -6u5       ��]�	8˰����A�*'

learning_rate_1o;

loss_1��@���5       ��]�	������A�*'

learning_rate_1o;

loss_1�j�@��#X5       ��]�	NϦ���A�*'

learning_rate_1o;

loss_1��@�c�r5       ��]�	5�ަ���A�*'

learning_rate_1o;

loss_1�Z�@;I�y5       ��]�	7������A�*'

learning_rate_1o;

loss_1J�@�� B5       ��]�	�T�����A�*'

learning_rate_1o;

loss_1L��@#S��5       ��]�	E�����A�*'

learning_rate_1o;

loss_1�n�@0vB�5       ��]�	M�����A�*'

learning_rate_1o;

loss_1]��@!-�5       ��]�	��*����A�*'

learning_rate_1o;

loss_1�3�@�c,o5       ��]�	�e:����A�*'

learning_rate_1o;

loss_1�|�@5XK5       ��]�	F�I����A�*'

learning_rate_1o;

loss_1C`�@��E�5       ��]�	�X����A�*'

learning_rate_1o;

loss_1�@��G5       ��]�	�Fh����A�*'

learning_rate_1o;

loss_1$�@�
n5       ��]�	dow����A�*'

learning_rate_1o;

loss_1���@��b5       ��]�	�������A�*'

learning_rate_1o;

loss_1�f�@K��5       ��]�	������A�*'

learning_rate_1o;

loss_16��@T��75       ��]�	lz�����A�*'

learning_rate_1o;

loss_1�/�@J�՚5       ��]�	�����A�*'

learning_rate_1o;

loss_1�d�@46F55       ��]�	�ħ���A�*'

learning_rate_1o;

loss_1f�@@��5       ��]�	��ӧ���A�*'

learning_rate_1o;

loss_1���@x��P5       ��]�	N~����A�*'

learning_rate_1o;

loss_1���@f�p5       ��]�	[�����A�*'

learning_rate_1o;

loss_1���@,�5       ��]�	k�����A�*'

learning_rate_1o;

loss_1��@.�}s5       ��]�	0����A�*'

learning_rate_1o;

loss_1�O�@fN55       ��]�	,� ����A�*'

learning_rate_1o;

loss_1�k�@+�BJ5       ��]�	m�/����A�*'

learning_rate_1o;

loss_1���@��45       ��]�	�I?����A�*'

learning_rate_1o;

loss_1���@r=�5       ��]�	_�N����A�*'

learning_rate_1o;

loss_1aT�@�՞n5       ��]�	�]����A�*'

learning_rate_1o;

loss_1���@a+�5       ��]�	@Jm����A�*'

learning_rate_1o;

loss_1��@F��5       ��]�	�|����A�*'

learning_rate_1o;

loss_1��@�D�95       ��]�	8ꋨ���A�*'

learning_rate_1o;

loss_1=P�@���5       ��]�	�M�����A�*'

learning_rate_1o;

loss_1M��@Ӛ� 5       ��]�	Ax�����A�*'

learning_rate_1o;

loss_1]�@p��15       ��]�	Bع����A�*'

learning_rate_1o;

loss_139�@����5       ��]�	�2ɨ���A�*'

learning_rate_1o;

loss_1Q��@b�a5       ��]�	psب���A�*'

learning_rate_1o;

loss_1���@Z�5       ��]�	h�����A�*'

learning_rate_1o;

loss_1���@g�5       ��]�	�������A�*'

learning_rate_1o;

loss_1֜�@9j��5       ��]�	�`����A�*'

learning_rate_1o;

loss_1���@S�>5       ��]�	������A�*'

learning_rate_1o;

loss_1l��@(;��5       ��]�	c%����A�*'

learning_rate_1o;

loss_1�>�@ |L5       ��]�	��4����A�*'

learning_rate_1o;

loss_1���@`��z5       ��]�	��C����A�*'

learning_rate_1o;

loss_1TW�@#O25       ��]�	�>S����A�*'

learning_rate_1o;

loss_1׀�@.v�5       ��]�	��b����A�*'

learning_rate_1o;

loss_1��@E{@5       ��]�	ɻq����A�*'

learning_rate_1o;

loss_1�E�@�[R�5       ��]�	�X�����A�*'

learning_rate_1o;

loss_1���@GB��5       ��]�	������A�*'

learning_rate_1o;

loss_1��@T E5       ��]�	�����A�*'

learning_rate_1o;

loss_1}�@<��45       ��]�	�������A�*'

learning_rate_1o;

loss_1ę�@6Z��5       ��]�	�־����A�*'

learning_rate_1o;

loss_1��@�v'�5       ��]�	6Ω���A�*'

learning_rate_1o;

loss_1{�@���M5       ��]�	x>ݩ���A�*'

learning_rate_1o;

loss_1=��@�tm5       ��]�	������A�*'

learning_rate_1o;

loss_1���@Τ��5       ��]�	z �����A�*'

learning_rate_1o;

loss_1_��@�=�5       ��]�	q#����A�*'

learning_rate_1o;

loss_1��@�|H5       ��]�	:X����A�*'

learning_rate_1o;

loss_1���@qC��5       ��]�	t�)����A�*'

learning_rate_1o;

loss_1���@�w��5       ��]�	�9����A�*'

learning_rate_1o;

loss_1C��@6��5       ��]�	,�H����A�*'

learning_rate_1o;

loss_1�#�@¡��5       ��]�	�W����A�*'

learning_rate_1o;

loss_1f��@(}�5       ��]�	��f����A�*'

learning_rate_1o;

loss_1a��@u�5       ��]�	�ev����A�*'

learning_rate_1o;

loss_1R5�@
�5       ��]�	i7�����A�*'

learning_rate_1o;

loss_1�6�@�bǩ5       ��]�	�������A�*'

learning_rate_1o;

loss_1X�@���5       ��]�	F������A�*'

learning_rate_1o;

loss_1��@����5       ��]�	;������A�*'

learning_rate_1o;

loss_1tC�@�l�5       ��]�	J�ê���A�*'

learning_rate_1o;

loss_1N)�@ �/�5       ��]�	%>Ӫ���A�*'

learning_rate_1o;

loss_1Z��@���T5       ��]�	������A�*'

learning_rate_1o;

loss_1�W�@��5�5       ��]�	v,����A�*'

learning_rate_1o;

loss_1{=�@�߉�5       ��]�	y�����A�*'

learning_rate_1o;

loss_1���@⟤5       ��]�	������A�*'

learning_rate_1o;

loss_1� �@Rʽ5       ��]�	�$ ����A�*'

learning_rate_1o;

loss_1;��@KA.�5       ��]�	Q/����A�*'

learning_rate_1o;

loss_1�@�~g5       ��]�	�y>����A�*'

learning_rate_1o;

loss_1&,�@ĸ�5       ��]�	�M����A�*'

learning_rate_1o;

loss_1��@[H�y5       ��]�	8]����A�*'

learning_rate_1o;

loss_1
��@&a%�5       ��]�	nil����A�*'

learning_rate_1o;

loss_1�Q�@�/�>5       ��]�	j�{����A�*'

learning_rate_1o;

loss_1@��@R9H�5       ��]�	�������A�*'

learning_rate_1o;

loss_1��@��5       ��]�	K�����A�*'

learning_rate_1o;

loss_1���@}0F5       ��]�	婫���A�*'

learning_rate_1o;

loss_1.�@Sv�F5       ��]�	������A�*'

learning_rate_1o;

loss_13.�@Xm��5       ��]�	a{ȫ���A�*'

learning_rate_1o;

loss_1T��@�H�5       ��]�	ְ׫���A�*'

learning_rate_1o;

loss_1���@�楯5       ��]�	%�����A�*'

learning_rate_1o;

loss_1-�@ZɃ45       ��]�	/@�����A�*'

learning_rate_1o;

loss_1���@��@5       ��]�	������A�*'

learning_rate_1o;

loss_1���@A�n5       ��]�	������A�*'

learning_rate_1o;

loss_1��@��$�5       ��]�	)$����A�*'

learning_rate_1o;

loss_1r<�@$�X)5       ��]�	�j3����A�*'

learning_rate_1o;

loss_1�Z�@��#F5       ��]�	܏B����A�*'

learning_rate_1o;

loss_1��@����5       ��]�	b�Q����A�*'

learning_rate_1o;

loss_1!$�@���5       ��]�	A(a����A�*'

learning_rate_1o;

loss_1n8�@�_�`5       ��]�	�Ap����A�*'

learning_rate_1o;

loss_1���@/�5       ��]�	�e����A�*'

learning_rate_1o;

loss_1F��@h. �5       ��]�	i�����A�*'

learning_rate_1o;

loss_1i��@\��5       ��]�	�������A�*'

learning_rate_1o;

loss_1i��@����5       ��]�	�٬����A�*'

learning_rate_1o;

loss_1v��@Ҙ_5       ��]�	|�����A�*'

learning_rate_1o;

loss_1���@��o5       ��]�	�Fˬ���A�*'

learning_rate_1o;

loss_1mm�@4F{K5       ��]�	�^ڬ���A�*'

learning_rate_1o;

loss_1�q�@�{�5       ��]�	������A�*'

learning_rate_1o;

loss_1�w�@�vI5       ��]�	�������A�*'

learning_rate_1o;

loss_1H�@5w�B5       ��]�	����A�*'

learning_rate_1o;

loss_1E��@7/$~5       ��]�	�U����A�*'

learning_rate_1o;

loss_1O��@n��5       ��]�	ɫ&����A�*'

learning_rate_1o;

loss_1���@�>5       ��]�	��5����A�*'

learning_rate_1o;

loss_1��@ 5       ��]�	|uE����A�*'

learning_rate_1o;

loss_1�l�@��]5       ��]�	�T����A�*'

learning_rate_1o;

loss_1E�@R^ת5       ��]�	��c����A�*'

learning_rate_1o;

loss_1���@d�W5       ��]�	`Ws����A�*'

learning_rate_1o;

loss_1�D�@y��5       ��]�	Ad�����A�*'

learning_rate_1o;

loss_1T[�@��T�5       ��]�	�ʑ����A�*'

learning_rate_1o;

loss_1�i�@tn5       ��]�	������A�*'

learning_rate_1o;

loss_1��@W��5       ��]�	Y������A�*'

learning_rate_1o;

loss_1���@�S՘5       ��]�	E������A�*'

learning_rate_1o;

loss_1�5�@~m�F5       ��]�	�	ϭ���A�*'

learning_rate_1o;

loss_1��@�xa5       ��]�	�ޭ���A�*'

learning_rate_1o;

loss_1ʗ�@�3F5       ��]�	"N�����A�*'

learning_rate_1o;

loss_1���@�WV�5       ��]�	K������A�*'

learning_rate_1o;

loss_1�)�@�1ލ5       ��]�	������A�*'

learning_rate_1o;

loss_1�5�@��,5       ��]�	������A�*'

learning_rate_1o;

loss_1���@�Bb5       ��]�	*++����A�*'

learning_rate_1o;

loss_1�N�@췯5       ��]�	~L:����A�*'

learning_rate_1o;

loss_1��@-��v5       ��]�	�I����A�*'

learning_rate_1o;

loss_1�W�@t)75       ��]�	T�X����A�*'

learning_rate_1o;

loss_1M9�@ɺ�:5       ��]�	th����A�*'

learning_rate_1o;

loss_1���@���5       ��]�	\ew����A�*'

learning_rate_1o;

loss_1���@h�	5       ��]�	������A�*'

learning_rate_1o;

loss_1�L�@,���5       ��]�	�]�����A�*'

learning_rate_1o;

loss_1���@���85       ��]�	������A�*'

learning_rate_1o;

loss_1���@�rL�5       ��]�	.�����A�*'

learning_rate_1o;

loss_1M��@C݄�5       ��]�	IwĮ���A�*'

learning_rate_1o;

loss_1nǻ@Je�5       ��]�	m�Ӯ���A�*'

learning_rate_1o;

loss_1_'�@��E�5       ��]�	� ����A�*'

learning_rate_1o;

loss_1��@$��5       ��]�	�/����A�*'

learning_rate_1o;

loss_1���@�<��5       ��]�	 �����A�*'

learning_rate_1o;

loss_1;N�@�F��5       ��]�	M�����A�*'

learning_rate_1o;

loss_1���@��5       ��]�	������A�*'

learning_rate_1o;

loss_1�B�@ J��5       ��]�	V/����A�*'

learning_rate_1o;

loss_1_��@O�؄5       ��]�	>>����A�*'

learning_rate_1o;

loss_1A��@�& �5       ��]�	�M����A�*'

learning_rate_1o;

loss_1\�@��5       ��]�	 �\����A�*'

learning_rate_1o;

loss_1�S�@mL�5       ��]�	��k����A�*'

learning_rate_1o;

loss_1�Ź@�T[p5       ��]�	J,{����A�*'

learning_rate_1o;

loss_1o�@b�5       ��]�	�z�����A�*'

learning_rate_1o;

loss_1(��@�q�5       ��]�	�������A�*'

learning_rate_1o;

loss_1���@͡/�5       ��]�	�Ĩ����A�*'

learning_rate_1o;

loss_1w��@.��5       ��]�	f7�����A�*'

learning_rate_1o;

loss_1T��@z�5       ��]�	ܪǯ���A�*'

learning_rate_1o;

loss_1T��@E��C5       ��]�	��֯���A�*'

learning_rate_1o;

loss_1���@A�/5       ��]�	�A����A�*'

learning_rate_1o;

loss_1 �@o���5       ��]�	2w�����A�*'

learning_rate_1o;

loss_1{d�@�A�t5       ��]�	3�����A�*'

learning_rate_1o;

loss_1Ľ@��g{5       ��]�	������A�*'

learning_rate_1o;

loss_1�q�@|��5       ��]�	��"����A�*'

learning_rate_1o;

loss_1S��@���5       ��]�	�^2����A�*'

learning_rate_1o;

loss_1t�@z�i5       ��]�	E�A����A�*'

learning_rate_1o;

loss_1$}�@���5       ��]�	t�P����A�*'

learning_rate_1o;

loss_1���@-J��5       ��]�	�P`����A�*'

learning_rate_1o;

loss_1i�@�|�5       ��]�	��o����A�*'

learning_rate_1o;

loss_1:к@�כ�5       ��]�	��~����A�*'

learning_rate_1o;

loss_1���@��m5       ��]�	h�����A�*'

learning_rate_1o;

loss_1���@��a�5       ��]�	������A�*'

learning_rate_1o;

loss_1-}�@��,�5       ��]�	W7�����A�*'

learning_rate_1o;

loss_1�ʶ@Ћ 5       ��]�	�̼����A�*'

learning_rate_1o;

loss_1�!�@�U�5       ��]�	�a̰���A�*'

learning_rate_1o;

loss_1�B�@Mt��5       ��]�	[ܰ���A�*'

learning_rate_1o;

loss_1潺@�*�5       ��]�	������A�*'

learning_rate_1o;

loss_1��@��G5       ��]�	-�����A�*'

learning_rate_1o;

loss_1D��@�Gj�5       ��]�	c}
����A�*'

learning_rate_1o;

loss_1r�@�=�5       ��]�	������A�*'

learning_rate_1o;

loss_1�;@�9�5       ��]�	s�(����A�*'

learning_rate_1o;

loss_1�@����5       ��]�	M<8����A�*'

learning_rate_1o;

loss_1�^�@�5       ��]�	G�G����A�*'

learning_rate_1o;

loss_1v~�@ޭ5�5       ��]�	f�V����A�*'

learning_rate_1o;

loss_1���@�]�5       ��]�	�f����A�*'

learning_rate_1o;

loss_1�D�@*5��5       ��]�	Zu����A�*'

learning_rate_1o;

loss_1�F�@L֯5       ��]�	�������A�*'

learning_rate_1o;

loss_1�"�@���5       ��]�	�������A�*'

learning_rate_1o;

loss_1���@���m5       ��]�	�!�����A�*'

learning_rate_1o;

loss_1�@��:�5       ��]�	�J�����A�*'

learning_rate_1o;

loss_1���@�#��5       ��]�	�������A�*'

learning_rate_1o;

loss_1�@�a_�5       ��]�	��б���A�*'

learning_rate_1o;

loss_1sq�@�u��5       ��]�	�����A�*'

learning_rate_1o;

loss_1�i�@A�t 5       ��]�	}u����A�*'

learning_rate_1o;

loss_1��@7�=5       ��]�	i������A�*'

learning_rate_1o;

loss_1��@#P�5       ��]�	Y�����A�*'

learning_rate_1o;

loss_1��@n;�,5       ��]�	�����A�*'

learning_rate_1o;

loss_1j��@�ӌ#5       ��]�	�,����A�*'

learning_rate_1o;

loss_1;��@�&�5       ��]�	��;����A�*'

learning_rate_1o;

loss_1�7�@CW�p5       ��]�	!�J����A�*'

learning_rate_1o;

loss_1@'�@n��5       ��]�	�;Z����A�*'

learning_rate_1o;

loss_1Fq�@N�DE5       ��]�	�3j����A�*'

learning_rate_1o;

loss_1�b�@	}�5       ��]�	K�z����A�*'

learning_rate_1o;

loss_1��@`��5       ��]�	%������A�*'

learning_rate_1o;

loss_1�I�@���	5       ��]�	)'�����A�*'

learning_rate_1o;

loss_1IL�@M>��5       ��]�	�N�����A�*'

learning_rate_1o;

loss_1���@P�5       ��]�	;ò���A�*'

learning_rate_1o;

loss_1e��@�5       ��]�	�<ֲ���A�*'

learning_rate_1o;

loss_1J*�@����5       ��]�	>����A�*'

learning_rate_1o;

loss_12�@�@��5       ��]�	������A�*'

learning_rate_1o;

loss_1�#�@�1�L5       ��]�	������A�*'

learning_rate_1o;

loss_1�C�@��l 5       ��]�	�n!����A�*'

learning_rate_1o;

loss_1���@UM�X5       ��]�	�4����A�*'

learning_rate_1o;

loss_1
ȼ@դB5       ��]�	�9G����A�*'

learning_rate_1o;

loss_1���@,
g45       ��]�	�~^����A�*'

learning_rate_1o;

loss_1�-�@k@5       ��]�	�r����A�*'

learning_rate_1o;

loss_1�I�@���?5       ��]�	F������A�*'

learning_rate_1o;

loss_1vD�@��a�5       ��]�	�P�����A�*'

learning_rate_1o;

loss_1��@��Ob5       ��]�	>������A�*'

learning_rate_1o;

loss_1�{�@�/E�5       ��]�	�³���A�*'

learning_rate_1o;

loss_1D-�@khJA5       ��]�	V�ֳ���A�*'

learning_rate_1o;

loss_1���@��ם5       ��]�	O�����A�*'

learning_rate_1o;

loss_1B��@���5       ��]�	w������A�*'

learning_rate_1o;

loss_1��@&s)�5       ��]�	������A�*'

learning_rate_1o;

loss_1��@�[��5       ��]�	�&����A�*'

learning_rate_1o;

loss_1�$�@���5       ��]�	��:����A�*'

learning_rate_1o;

loss_1���@��ĵ5       ��]�	��N����A�*'

learning_rate_1o;

loss_1���@�(�5       ��]�	��b����A�*'

learning_rate_1o;

loss_1��@T���5       ��]�	b�v����A�*'

learning_rate_1o;

loss_1Q.�@���S5       ��]�	4ኴ���A�*'

learning_rate_1o;

loss_1p��@x�T�5       ��]�	Ž�����A�*'

learning_rate_1o;

loss_1���@�5�5       ��]�	a~�����A�*'

learning_rate_1o;

loss_1y��@8�v�5       ��]�	�Aƴ���A�*'

learning_rate_1o;

loss_1��@S@Y�5       ��]�	ϣڴ���A�*'

learning_rate_1o;

loss_1~�@���5       ��]�	�x����A�*'

learning_rate_1o;

loss_1z��@��dT5       ��]�	�y����A�*'

learning_rate_1o;

loss_1��@�Zμ5       ��]�	D����A�*'

learning_rate_1o;

loss_1ʃ�@�]�5       ��]�	u�*����A�*'

learning_rate_1o;

loss_1:d�@H.�"5       ��]�	ȼ>����A�*'

learning_rate_1o;

loss_1��@Ӄ�=5       ��]�	�R����A�*'

learning_rate_1o;

loss_1B��@g6,l5       ��]�	Ipf����A�*'

learning_rate_1o;

loss_1Y�@9�Z5       ��]�	�4z����A�*'

learning_rate_1o;

loss_1�\�@oPX|5       ��]�	fs�����A�*'

learning_rate_1o;

loss_1�@�@�(�5       ��]�	�;�����A�*'

learning_rate_1o;

loss_1���@��h�5       ��]�	������A�*'

learning_rate_1o;

loss_1�6�@E��5       ��]�	�ʵ���A�*'

learning_rate_1o;

loss_1��@��Z5       ��]�	qv޵���A�*'

learning_rate_1o;

loss_1���@n߼�5       ��]�	ck����A�*'

learning_rate_1o;

loss_1�_�@��%5       ��]�	*|����A�*'

learning_rate_1o;

loss_1�9�@/N�g5       ��]�	�c����A�*'

learning_rate_1o;

loss_1��@?T�5       ��]�	�h.����A�*'

learning_rate_1o;

loss_1&!�@0�4 5       ��]�	�/B����A�*'

learning_rate_1o;

loss_10ҿ@�^5       ��]�	�^V����A�*'

learning_rate_1o;

loss_1�^�@�_y5       ��]�	�wj����A�*'

learning_rate_1o;

loss_1+!�@���p5       ��]�	k�~����A�*'

learning_rate_1o;

loss_1��@�Ov�5       ��]�	ɒ�����A�*'

learning_rate_1o;

loss_1�-�@�9�5       ��]�	`a�����A�*'

learning_rate_1o;

loss_1_�@m��5       ��]�	�~�����A�*'

learning_rate_1o;

loss_1��@�)t5       ��]�	L{ζ���A�*'

learning_rate_1o;

loss_1-�@zd�5       ��]�	U[����A�*'

learning_rate_1o;

loss_1ͺ�@r.�L5       ��]�	&|�����A�*'

learning_rate_1o;

loss_12��@Ї��5       ��]�	 <
����A�*'

learning_rate_1o;

loss_1�o�@��$5       ��]�	������A�*'

learning_rate_1o;

loss_1��@���5       ��]�	��1����A�*'

learning_rate_1o;

loss_1a��@�g5       ��]�	�E����A�*'

learning_rate_1o;

loss_1���@� ��5       ��]�	 VY����A�*'

learning_rate_1o;

loss_10��@!�W&5       ��]�	��m����A�*'

learning_rate_1o;

loss_1b:�@A�55       ��]�	�������A�*'

learning_rate_1o;

loss_1Ӏ�@�S5       ��]�	정����A�*'

learning_rate_1o;

loss_1x:�@;t�5       ��]�	lͩ����A�*'

learning_rate_1o;

loss_1�@@Z�=5       ��]�	������A�*'

learning_rate_1o;

loss_10<�@u}�~5       ��]�	j�ѷ���A�*'

learning_rate_1o;

loss_1XZ�@a���5       ��]�	������A�*'

learning_rate_1o;

loss_1O�@�<\5       ��]�	�������A�*'

learning_rate_1o;

loss_1��@V45       ��]�	.�����A�*'

learning_rate_1o;

loss_1�9�@�*5       ��]�	S�!����A�*'

learning_rate_1o;

loss_1�Ǻ@%�4Q5       ��]�	�5����A�*'

learning_rate_1o;

loss_1t,�@�w&p5       ��]�	�I����A�*'

learning_rate_1o;

loss_1�Խ@��N5       ��]�	��]����A�*'

learning_rate_1o;

loss_1c��@:N�5       ��]�	K�q����A�*'

learning_rate_1o;

loss_1�8�@�41$5       ��]�	]������A�*'

learning_rate_1o;

loss_1���@^k��5       ��]�	5������A�*'

learning_rate_1o;

loss_1���@ �Z5       ��]�	GЭ����A�*'

learning_rate_1o;

loss_1)��@ݭ	T5       ��]�	7¸���A�*'

learning_rate_1o;

loss_16l�@ E��5       ��]�	_�ո���A�*'

learning_rate_1o;

loss_1Ū�@�R�R5       ��]�	l�����A�*'

learning_rate_1o;

loss_12��@Wrh�5       ��]�	Q������A�*'

learning_rate_1o;

loss_1�H�@{^P5       ��]�	������A�*'

learning_rate_1o;

loss_1N��@�G�5       ��]�	�%����A�*'

learning_rate_1o;

loss_1p�@U�t�5       ��]�	P�9����A�*'

learning_rate_1o;

loss_1���@+n/)5       ��]�	��M����A�*'

learning_rate_1o;

loss_1���@O!k|5       ��]�	�b����A�*'

learning_rate_1o;

loss_1��@����5       ��]�	�Ev����A�*'

learning_rate_1o;

loss_1:��@I���5       ��]�	J(�����A�*'

learning_rate_1o;

loss_1|�@sU25       ��]�	�N�����A�*'

learning_rate_1o;

loss_1��@9���5       ��]�	�[�����A�*'

learning_rate_1o;

loss_1[��@���5       ��]�	�oƹ���A�*'

learning_rate_1o;

loss_1��@�h��5       ��]�	^ڹ���A�*'

learning_rate_1o;

loss_1�r�@{9;<5       ��]�	CR����A�*'

learning_rate_1o;

loss_1a��@H�y�5       ��]�	�����A�*'

learning_rate_1o;

loss_1�u�@�ѕ5       ��]�	����A�*'

learning_rate_1o;

loss_1喻@Zn�5       ��]�	d�+����A�*'

learning_rate_1o;

loss_1#K�@ƾ5       ��]�	��?����A�*'

learning_rate_1o;

loss_1���@�^��5       ��]�	��S����A�*'

learning_rate_1o;

loss_1�*�@��6�5       ��]�	�Dg����A�*'

learning_rate_1o;

loss_1-�@�L�S5       ��]�	�'{����A�*'

learning_rate_1o;

loss_1B"�@�ؚF5       ��]�	������A�*'

learning_rate_1o;

loss_1�f�@�h5       ��]�	�d�����A�*'

learning_rate_1o;

loss_1�w�@���5       ��]�	{Z�����A�*'

learning_rate_1o;

loss_1���@��r5       ��]�	�b˺���A�*'

learning_rate_1o;

loss_1.d�@(E^�5       ��]�	MTߺ���A�*'

learning_rate_1o;

loss_1���@(�uv5       ��]�	WL����A�*'

learning_rate_1o;

loss_1e�@���5       ��]�	�P����A�*'

learning_rate_1o;

loss_1h�@-�׏5       ��]�	ړ����A�*'

learning_rate_1o;

loss_1���@�Ī�5       ��]�	R|/����A�*'

learning_rate_1o;

loss_1[��@�E�5       ��]�	��C����A�*'

learning_rate_1o;

loss_1�\�@��5       ��]�	��W����A�*'

learning_rate_1o;

loss_1tǾ@���5       ��]�	��k����A�*'

learning_rate_1o;

loss_1�P�@��T5       ��]�	�|����A�*'

learning_rate_1o;

loss_1���@�,t�5       ��]�	�u�����A�*'

learning_rate_1o;

loss_1��@a�$�5       ��]�	�]�����A�*'

learning_rate_1o;

loss_1�@����5       ��]�	�R�����A�*'

learning_rate_1o;

loss_1�O�@���45       ��]�	S ϻ���A�*'

learning_rate_1o;

loss_1p�@\���5       ��]�	L�����A�*'

learning_rate_1o;

loss_1��@�o5       ��]�	"������A�*'

learning_rate_1o;

loss_1S��@Iķ�5       ��]�	��
����A�*'

learning_rate_1o;

loss_1�c�@�L�H5       ��]�	�c����A�*'

learning_rate_1o;

loss_1¸�@�f��5       ��]�	n�2����A�*'

learning_rate_1o;

loss_1���@�-5       ��]�	��F����A�*'

learning_rate_1o;

loss_1k��@+��5       ��]�	`�Z����A�*'

learning_rate_1o;

loss_1]��@k�a5       ��]�	��n����A�*'

learning_rate_1o;

loss_1�B�@��1�5       ��]�	�����A�*'

learning_rate_1o;

loss_1���@s��_5       ��]�	O;�����A�*'

learning_rate_1o;

loss_1�3�@a:F-5       ��]�	Cΰ����A�*'

learning_rate_1o;

loss_1���@���5       ��]�	�!Ǽ���A�*'

learning_rate_1o;

loss_1���@�I��5       ��]�	�5ݼ���A�*'

learning_rate_1o;

loss_1c��@	�5       ��]�	(�����A�*'

learning_rate_1o;

loss_1��@u�U5       ��]�	��	����A�*'

learning_rate_1o;

loss_1C��@�<D�5       ��]�	�c ����A�*'

learning_rate_1o;

loss_1���@�T�5       ��]�	�6����A�*'

learning_rate_1o;

loss_1���@F���5       ��]�	]M����A�*'

learning_rate_1o;

loss_1���@1��i5       ��]�	bQc����A�*'

learning_rate_1o;

loss_1�@t�K5       ��]�	{y����A�*'

learning_rate_1o;

loss_1���@<{�5       ��]�	(�����A�*'

learning_rate_1o;

loss_17m�@F{5       ��]�	�L�����A�*'

learning_rate_1o;

loss_1�z�@m�O5       ��]�	[ż����A�*'

learning_rate_1o;

loss_1롿@��S�5       ��]�	6ӽ���A�*'

learning_rate_1o;

loss_1�m�@o�k5       ��]�	�{����A�*'

learning_rate_1o;

loss_1W��@�Fo5       ��]�	R������A�*'

learning_rate_1o;

loss_1�D�@�/q5       ��]�	�����A�*'

learning_rate_1o;

loss_1��@&��5       ��]�	&�+����A�*'

learning_rate_1o;

loss_1��@Sѽo5       ��]�	��A����A�*'

learning_rate_1o;

loss_1���@zX��5       ��]�	4X����A�*'

learning_rate_1o;

loss_1\��@��U�5       ��]�	N�n����A�*'

learning_rate_1o;

loss_1�5�@;�q�5       ��]�	F�����A�*'

learning_rate_1o;

loss_1#e�@X�}55       ��]�	�d�����A�*'

learning_rate_1o;

loss_1���@����5       ��]�	�������A�*'

learning_rate_1o;

loss_1��@˷�5       ��]�	"JȾ���A�*'

learning_rate_1o;

loss_1r�@?uSP5       ��]�	m�޾���A�*'

learning_rate_1o;

loss_1���@�x��5       ��]�	!������A�*'

learning_rate_1o;

loss_1���@��5       ��]�	�����A�*'

learning_rate_1o;

loss_1��@�=�@5       ��]�	�T!����A�*'

learning_rate_1o;

loss_1���@�0|5       ��]�	��7����A�*'

learning_rate_1o;

loss_1���@�h�+5       ��]�	��M����A�*'

learning_rate_1o;

loss_1�i�@�c�5       ��]�	�Pd����A�*'

learning_rate_1o;

loss_1��@{'��5       ��]�	�m{����A�*'

learning_rate_1o;

loss_1��@�E�5       ��]�	@������A�*'

learning_rate_1o;

loss_1C�@n��95       ��]�	����A�*'

learning_rate_1o;

loss_1-
�@��955       ��]�	�)�����A�*'

learning_rate_1o;

loss_1�f�@b�Z5       ��]�	��Կ���A�*'

learning_rate_1o;

loss_1�<�@�5�
5       ��]�	�U�����A�*'

learning_rate_1o;

loss_1* �@���5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1�L�@���5       ��]�	�"����A�*'

learning_rate_1�G�:

loss_1�B�@�?��5       ��]�	9����A�*'

learning_rate_1�G�:

loss_1��@MaF�5       ��]�	EO����A�*'

learning_rate_1�G�:

loss_19��@��q/5       ��]�	�f����A�*'

learning_rate_1�G�:

loss_1i��@�$O�5       ��]�	��|����A�*'

learning_rate_1�G�:

loss_1���@��O�5       ��]�	�v�����A�*'

learning_rate_1�G�:

loss_1eT�@$��5       ��]�	q������A�*'

learning_rate_1�G�:

loss_1���@w:Y5       ��]�	(�����A�*'

learning_rate_1�G�:

loss_1���@NY�05       ��]�	}������A�*'

learning_rate_1�G�:

loss_1���@@�5       ��]�	=������A�*'

learning_rate_1�G�:

loss_1f|�@vg�5       ��]�	,4����A�*'

learning_rate_1�G�:

loss_1c��@as�x5       ��]�	�[����A�*'

learning_rate_1�G�:

loss_1�t�@A�$�5       ��]�	ϭ2����A�*'

learning_rate_1�G�:

loss_1}�@")�5       ��]�	I����A�*'

learning_rate_1�G�:

loss_1ɘ�@�]vs5       ��]�	l{_����A�*'

learning_rate_1�G�:

loss_1�G�@CX<�5       ��]�	�u����A�*'

learning_rate_1�G�:

loss_1Jx�@�S�@5       ��]�	�3�����A�*'

learning_rate_1�G�:

loss_1ao�@1���5       ��]�	�{�����A�*'

learning_rate_1�G�:

loss_1La�@�7p�5       ��]�	�e�����A�*'

learning_rate_1�G�:

loss_1���@�.�&5       ��]�	U�����A�*'

learning_rate_1�G�:

loss_1�9�@zG��5       ��]�	+�����A�*'

learning_rate_1�G�:

loss_1%v�@%3O�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�-�@VM�5       ��]�	S4����A�*'

learning_rate_1�G�:

loss_1�ԓ@o�#b5       ��]�	%�*����A�*'

learning_rate_1�G�:

loss_1X��@ϑ�5       ��]�	Q,B����A�*'

learning_rate_1�G�:

loss_1�͉@�K�5       ��]�	ςX����A�*'

learning_rate_1�G�:

loss_1UE�@��v�5       ��]�	��n����A�*'

learning_rate_1�G�:

loss_1�_�@	M�5       ��]�	�Ǆ����A�*'

learning_rate_1�G�:

loss_10�@�r��5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1P�@�>�5       ��]�	�߲����A�*'

learning_rate_1�G�:

loss_1�ˊ@�q#�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1֣�@� �5       ��]�	*�����A�*'

learning_rate_1�G�:

loss_1UR�@9w��5       ��]�	)8�����A�*'

learning_rate_1�G�:

loss_1w�@�QD�5       ��]�	-����A�*'

learning_rate_1�G�:

loss_1�փ@���z5       ��]�	�d#����A�*'

learning_rate_1�G�:

loss_1���@���'5       ��]�	>�9����A�*'

learning_rate_1�G�:

loss_1�|@_	��5       ��]�	��O����A�*'

learning_rate_1�G�:

loss_1�@-�05       ��]�	[�f����A�*'

learning_rate_1�G�:

loss_1��@�<�Q5       ��]�	F}����A�*'

learning_rate_1�G�:

loss_1<:�@��t!5       ��]�	U�����A�*'

learning_rate_1�G�:

loss_1g��@��>�5       ��]�	]�����A�*'

learning_rate_1�G�:

loss_1��v@?.5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1���@�̀�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1$�u@ѷ�(5       ��]�	�������A�*'

learning_rate_1�G�:

loss_15��@V�W5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�Xk@��*-5       ��]�	�1����A�*'

learning_rate_1�G�:

loss_1���@4{a-5       ��]�	��2����A�*'

learning_rate_1�G�:

loss_1t�@RT#75       ��]�	n�I����A�*'

learning_rate_1�G�:

loss_1C|@���&5       ��]�	
�`����A�*'

learning_rate_1�G�:

loss_1�yw@�-�5       ��]�	�w����A�*'

learning_rate_1�G�:

loss_1��@s��5       ��]�	�}�����A�*'

learning_rate_1�G�:

loss_15��@�p��5       ��]�	>�����A�*'

learning_rate_1�G�:

loss_1;hq@1+��5       ��]�	�ǻ����A�*'

learning_rate_1�G�:

loss_1r@E�r�5       ��]�	�.�����A�*'

learning_rate_1�G�:

loss_1���@V�2]5       ��]�	�c�����A�*'

learning_rate_1�G�:

loss_1FA�@�]�Z5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1��z@8�6�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��@m��(5       ��]�	��+����A�*'

learning_rate_1�G�:

loss_1
_q@�/��5       ��]�	B����A�*'

learning_rate_1�G�:

loss_1E�@ ��5       ��]�	_6Z����A�*'

learning_rate_1�G�:

loss_1Z�f@k��5       ��]�	�ep����A�*'

learning_rate_1�G�:

loss_1�v@GUH55       ��]�	k�����A�*'

learning_rate_1�G�:

loss_1-�s@i�
�5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1�t@�Z��5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1��t@>��5       ��]�	�F�����A�*'

learning_rate_1�G�:

loss_1*�@���5       ��]�	CS�����A�*'

learning_rate_1�G�:

loss_1pXz@/)�5       ��]�	�x�����A�*'

learning_rate_1�G�:

loss_1d�o@��S]5       ��]�	#�����A�*'

learning_rate_1�G�:

loss_1n�t@,���5       ��]�	S�$����A�*'

learning_rate_1�G�:

loss_1͑{@��ְ5       ��]�	lI;����A�*'

learning_rate_1�G�:

loss_17c@�Ƴ5       ��]�	k�Q����A�*'

learning_rate_1�G�:

loss_19�@g^��5       ��]�	�\i����A�*'

learning_rate_1�G�:

loss_1d1m@���5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1�Hq@z�W�5       ��]�	:[�����A�*'

learning_rate_1�G�:

loss_1P"u@��n�5       ��]�	�n�����A�*'

learning_rate_1�G�:

loss_1��x@ޑ,b5       ��]�	�p�����A�*'

learning_rate_1�G�:

loss_1�/}@��Y|5       ��]�	i}�����A�*'

learning_rate_1�G�:

loss_1;s{@�u�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�^n@�Y��5       ��]�	�p����A�*'

learning_rate_1�G�:

loss_1;�q@�/�5       ��]�	L�����A�*'

learning_rate_1�G�:

loss_1��q@D��5       ��]�	�$4����A�*'

learning_rate_1�G�:

loss_1�o@��=5       ��]�	�VJ����A�*'

learning_rate_1�G�:

loss_1��x@�&]5       ��]�	�`����A�*'

learning_rate_1�G�:

loss_1�ip@x��5       ��]�	��v����A�*'

learning_rate_1�G�:

loss_1L�@�ɩ5       ��]�	6�����A�*'

learning_rate_1�G�:

loss_1j�~@�qA=5       ��]�	������A�*'

learning_rate_1�G�:

loss_1+�l@�E��5       ��]�	ɚ�����A�*'

learning_rate_1�G�:

loss_1�U{@���5       ��]�	b������A�*'

learning_rate_1�G�:

loss_1v1~@�U85       ��]�	f������A�*'

learning_rate_1�G�:

loss_1� v@����5       ��]�	pD�����A�*'

learning_rate_1�G�:

loss_1�yx@�lU25       ��]�	�p����A�*'

learning_rate_1�G�:

loss_1�q@�l�5       ��]�	b*����A�*'

learning_rate_1�G�:

loss_1��g@d��5       ��]�	�C@����A�*'

learning_rate_1�G�:

loss_1C�t@�j�5       ��]�	}�V����A�*'

learning_rate_1�G�:

loss_1��m@I��G5       ��]�	��l����A�*'

learning_rate_1�G�:

loss_1m?i@_?1K5       ��]�	o������A�*'

learning_rate_1�G�:

loss_1��w@��i5       ��]�	�X�����A�*'

learning_rate_1�G�:

loss_1��i@Nn�{5       ��]�	"������A�*'

learning_rate_1�G�:

loss_1��x@�ˠ�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1d׃@U���5       ��]�	9������A�*'

learning_rate_1�G�:

loss_1ʈ@.D�5       ��]�	I������A�*'

learning_rate_1�G�:

loss_1�l@(eI�5       ��]�	r�
����A�*'

learning_rate_1�G�:

loss_1��|@��KX5       ��]�	�4!����A�*'

learning_rate_1�G�:

loss_1��@�r��5       ��]�	�P7����A�*'

learning_rate_1�G�:

loss_10o@0�(5       ��]�	��M����A�*'

learning_rate_1�G�:

loss_1Љw@��r�5       ��]�	�d����A�*'

learning_rate_1�G�:

loss_1a2�@O��5       ��]�	^�z����A�*'

learning_rate_1�G�:

loss_1k�w@<&5       ��]�	�e�����A�*'

learning_rate_1�G�:

loss_1T�`@�Z(�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�x}@ֹ,�5       ��]�	xs�����A�*'

learning_rate_1�G�:

loss_1}f@�P_d5       ��]�	�8�����A�*'

learning_rate_1�G�:

loss_1vCg@�Y�c5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1
�r@���5       ��]�	�J����A�*'

learning_rate_1�G�:

loss_1�t@j��75       ��]�	�+����A�*'

learning_rate_1�G�:

loss_1cN�@����5       ��]�	6�2����A�*'

learning_rate_1�G�:

loss_1�Xs@���y5       ��]�	{I����A�*'

learning_rate_1�G�:

loss_1��r@�j5       ��]�	�#`����A�*'

learning_rate_1�G�:

loss_1�o�@����5       ��]�	�nv����A�*'

learning_rate_1�G�:

loss_1�Xh@`�ʪ5       ��]�	�ˌ����A�*'

learning_rate_1�G�:

loss_1�k@����5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1p�t@�85       ��]�	Ch�����A�*'

learning_rate_1�G�:

loss_1fn@gO5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1��w@u9n5       ��]�	������A�*'

learning_rate_1�G�:

loss_1ؕ�@�ؓ�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1vIl@����5       ��]�	[�����A�*'

learning_rate_1�G�:

loss_1��j@د�5       ��]�	!Y)����A�*'

learning_rate_1�G�:

loss_1{�{@�C�`5       ��]�		�?����A�*'

learning_rate_1�G�:

loss_1��o@���5       ��]�	��U����A�*'

learning_rate_1�G�:

loss_1�k|@��5       ��]�	z�l����A�*'

learning_rate_1�G�:

loss_1��{@I��}5       ��]�	�ɂ����A�*'

learning_rate_1�G�:

loss_1\1h@�q85       ��]�	iB�����A�*'

learning_rate_1�G�:

loss_1cp@}�ր5       ��]�	i������A�*'

learning_rate_1�G�:

loss_1���@J.}�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��y@7��5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�-|@�X5       ��]�	D������A�*'

learning_rate_1�G�:

loss_1��q@R��5       ��]�	Gs
����A�*'

learning_rate_1�G�:

loss_1s�t@G�5       ��]�	s� ����A�*'

learning_rate_1�G�:

loss_1�;v@����5       ��]�	�6����A�*'

learning_rate_1�G�:

loss_1�m@��a5       ��]�	�iM����A�*'

learning_rate_1�G�:

loss_1Cq@m��5       ��]�	B�c����A�*'

learning_rate_1�G�:

loss_1�Md@آ;�5       ��]�		�y����A�*'

learning_rate_1�G�:

loss_1�{i@��K�5       ��]�	TϏ����A�*'

learning_rate_1�G�:

loss_1�m@�<:5       ��]�	_�����A�*'

learning_rate_1�G�:

loss_1o(k@Z��P5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�.h@��O5       ��]�	=������A�*'

learning_rate_1�G�:

loss_1�!@f�+5       ��]�	'�����A�*'

learning_rate_1�G�:

loss_1�tn@���5       ��]�	������A�*'

learning_rate_1�G�:

loss_1
v@h�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��y@�5��5       ��]�	,3/����A�*'

learning_rate_1�G�:

loss_1y�^@c��=5       ��]�	�F����A�*'

learning_rate_1�G�:

loss_1\hi@3�j�5       ��]�	r]����A�*'

learning_rate_1�G�:

loss_1��g@'�.m5       ��]�	U�s����A�*'

learning_rate_1�G�:

loss_1#�t@pc.x5       ��]�	qۉ����A�*'

learning_rate_1�G�:

loss_1��x@z<D�5       ��]�	1i�����A�*'

learning_rate_1�G�:

loss_1�r�@6��5       ��]�	$������A�*'

learning_rate_1�G�:

loss_1r�v@�m3�5       ��]�	x:�����A�*'

learning_rate_1�G�:

loss_1�pj@d�e�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1&�@��35       ��]�	r������A�*'

learning_rate_1�G�:

loss_1��w@��H:5       ��]�	�*����A�*'

learning_rate_1�G�:

loss_19n@�555       ��]�	��&����A�*'

learning_rate_1�G�:

loss_1]bu@/=�5       ��]�	Q�<����A�*'

learning_rate_1�G�:

loss_1�~@���}5       ��]�	-3S����A�*'

learning_rate_1�G�:

loss_1-�j@M��5       ��]�	�<k����A�*'

learning_rate_1�G�:

loss_1�Ni@���5       ��]�	Q�����A�*'

learning_rate_1�G�:

loss_1�m@y�j�5       ��]�	�
�����A�*'

learning_rate_1�G�:

loss_1B�c@���5       ��]�	,�����A�*'

learning_rate_1�G�:

loss_1�x@ ���5       ��]�	������A�*'

learning_rate_1�G�:

loss_1Ӊu@y>�5       ��]�	� �����A�*'

learning_rate_1�G�:

loss_1��m@��U5       ��]�	am�����A�*'

learning_rate_1�G�:

loss_1ɐw@�*%�5       ��]�	�	����A�*'

learning_rate_1�G�:

loss_1p@B!i�5       ��]�	A�����A�*'

learning_rate_1�G�:

loss_10�n@��=b5       ��]�	��7����A�*'

learning_rate_1�G�:

loss_1��t@����5       ��]�	��M����A�*'

learning_rate_1�G�:

loss_1cCr@v>sw5       ��]�	�$d����A�*'

learning_rate_1�G�:

loss_1}�q@AO�5       ��]�	�|z����A�*'

learning_rate_1�G�:

loss_1�\w@$��5       ��]�	bא����A�*'

learning_rate_1�G�:

loss_1�o@�v��5       ��]�	������A�*'

learning_rate_1�G�:

loss_1kw@_Ez�5       ��]�	�-�����A�*'

learning_rate_1�G�:

loss_1��q@� �5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�u@��4�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�R@���5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�c@��}�5       ��]�	?n����A�*'

learning_rate_1�G�:

loss_1h�u@=oO5       ��]�	�1����A�*'

learning_rate_1�G�:

loss_1h�i@��y�5       ��]�	\�G����A�*'

learning_rate_1�G�:

loss_1Рr@��T�5       ��]�	6�^����A�*'

learning_rate_1�G�:

loss_1�y@
�[�5       ��]�	�u����A�*'

learning_rate_1�G�:

loss_1: �@��r5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�u@�5       ��]�	4������A�*'

learning_rate_1�G�:

loss_12-e@���5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�b�@/ϑw5       ��]�	������A�*'

learning_rate_1�G�:

loss_1;�j@p���5       ��]�	MF�����A�*'

learning_rate_1�G�:

loss_1�g@�`Y5       ��]�	^c�����A�*'

learning_rate_1�G�:

loss_1��v@`��u5       ��]�	R����A�*'

learning_rate_1�G�:

loss_1�}l@|+5       ��]�	{K)����A�*'

learning_rate_1�G�:

loss_1J�n@��&g5       ��]�	J�?����A�*'

learning_rate_1�G�:

loss_1{�\@Y;�5       ��]�	�V����A�*'

learning_rate_1�G�:

loss_19�s@��˽5       ��]�	�	m����A�*'

learning_rate_1�G�:

loss_1��f@`6� 5       ��]�	}c�����A�*'

learning_rate_1�G�:

loss_1]ms@�{*�5       ��]�	e������A�*'

learning_rate_1�G�:

loss_1�zv@����5       ��]�	�4�����A�*'

learning_rate_1�G�:

loss_1�a@vjj5       ��]�	�x�����A�*'

learning_rate_1�G�:

loss_1hNt@ogj�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1Z[~@�^,5       ��]�	{$�����A�*'

learning_rate_1�G�:

loss_10`m@�z�5       ��]�	�g
����A�*'

learning_rate_1�G�:

loss_1��k@9�5       ��]�	E� ����A�*'

learning_rate_1�G�:

loss_14zu@�# �5       ��]�	Q�8����A�*'

learning_rate_1�G�:

loss_1�2x@�p�m5       ��]�	��N����A�*'

learning_rate_1�G�:

loss_1x��@�@��5       ��]�	�d����A�*'

learning_rate_1�G�:

loss_1h�x@���5       ��]�	z8{����A�*'

learning_rate_1�G�:

loss_1<H�@r���5       ��]�	"�����A�*'

learning_rate_1�G�:

loss_1�{@���5       ��]�	Y������A�*'

learning_rate_1�G�:

loss_1��{@�x�A5       ��]�	<�����A�*'

learning_rate_1�G�:

loss_1Ee@5t�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1Ve@ǷY\5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1}�@��Q5       ��]�	 �����A�*'

learning_rate_1�G�:

loss_1��o@;���5       ��]�	~����A�*'

learning_rate_1�G�:

loss_1X�o@�?��5       ��]�	q�/����A�*'

learning_rate_1�G�:

loss_1�bn@ѿ��5       ��]�	�eG����A�*'

learning_rate_1�G�:

loss_1�h^@����5       ��]�	�_����A�*'

learning_rate_1�G�:

loss_1�Di@q��5       ��]�	Du����A�*'

learning_rate_1�G�:

loss_1�m@O��5       ��]�	������A�*'

learning_rate_1�G�:

loss_10w@>�CI5       ��]�	�n�����A�*'

learning_rate_1�G�:

loss_1Zc|@S/�#5       ��]�	}������A�*'

learning_rate_1�G�:

loss_1�!w@lIQ)5       ��]�	d������A�*'

learning_rate_1�G�:

loss_1�nt@��@�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1\+t@��A5       ��]�	������A�*'

learning_rate_1�G�:

loss_1&�q@ �f�5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1�B{@,�$5       ��]�	L�'����A�*'

learning_rate_1�G�:

loss_1
�r@��ż5       ��]�	%M=����A�*'

learning_rate_1�G�:

loss_1*!r@4�ӑ5       ��]�	$QS����A�*'

learning_rate_1�G�:

loss_1�k�@K�Rh5       ��]�	Cj����A�*'

learning_rate_1�G�:

loss_1���@[+x5       ��]�	?o�����A�*'

learning_rate_1�G�:

loss_1V�q@��5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�nn@pݞr5       ��]�	>B�����A�*'

learning_rate_1�G�:

loss_1Ռx@DG�5       ��]�	�G�����A�*'

learning_rate_1�G�:

loss_1>�m@oV5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1C�y@�#��5       ��]�	&������A�*'

learning_rate_1�G�:

loss_10t@���`5       ��]�	������A�*'

learning_rate_1�G�:

loss_1>�r@?D�_5       ��]�	b�����A�*'

learning_rate_1�G�:

loss_1�u@��C5       ��]�	�!2����A�*'

learning_rate_1�G�:

loss_1гm@|��5       ��]�	Y�I����A�*'

learning_rate_1�G�:

loss_1��f@ݐ�5       ��]�	�_����A�*'

learning_rate_1�G�:

loss_1Ǳl@�V��5       ��]�	-mv����A�*'

learning_rate_1�G�:

loss_1Ίx@�ƃ�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1;o@�֟�5       ��]�	�L�����A�*'

learning_rate_1�G�:

loss_16��@����5       ��]�	Ź����A�*'

learning_rate_1�G�:

loss_1�b@���5       ��]�	\������A�*'

learning_rate_1�G�:

loss_1֤@��/�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1� l@]ue5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1\�y@���f5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1P�t@3�J�5       ��]�	��(����A�*'

learning_rate_1�G�:

loss_1p�[@N�W5       ��]�	�4?����A�*'

learning_rate_1�G�:

loss_1�\a@s#5       ��]�	�wU����A�*'

learning_rate_1�G�:

loss_1�k@�b"5       ��]�	k/l����A�*'

learning_rate_1�G�:

loss_1%\@�դf5       ��]�	T�����A�*'

learning_rate_1�G�:

loss_1��v@茌�5       ��]�	�4�����A�*'

learning_rate_1�G�:

loss_1��p@�	�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�jn@�
�X5       ��]�	ji�����A�*'

learning_rate_1�G�:

loss_1�^@" 5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1��i@�K?5       ��]�	�(�����A�*'

learning_rate_1�G�:

loss_1� b@���5       ��]�	��
����A�*'

learning_rate_1�G�:

loss_1�&f@u�5       ��]�	�E!����A�*'

learning_rate_1�G�:

loss_1��s@�L�5       ��]�	kk7����A�*'

learning_rate_1�G�:

loss_1;�b@ W��5       ��]�	��M����A�*'

learning_rate_1�G�:

loss_1�n@��&5       ��]�	d����A�*'

learning_rate_1�G�:

loss_1m�h@�r35       ��]�	2cz����A�*'

learning_rate_1�G�:

loss_1Zwn@\��5       ��]�	�f�����A�*'

learning_rate_1�G�:

loss_1�Eq@��'�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��@��΁5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�.c@�ӡ�5       ��]�	�N�����A�*'

learning_rate_1�G�:

loss_1=t@��+5       ��]�	�l�����A�*'

learning_rate_1�G�:

loss_1_k@�T�<5       ��]�	�� ����A�*'

learning_rate_1�G�:

loss_1k�]@�h �5       ��]�	�'����A�*'

learning_rate_1�G�:

loss_1Z�c@]��5       ��]�	yQ-����A�*'

learning_rate_1�G�:

loss_1��n@}ЂV5       ��]�	3�C����A�*'

learning_rate_1�G�:

loss_1ps@��EJ5       ��]�	Z����A�*'

learning_rate_1�G�:

loss_1}}k@S�T�5       ��]�	+�p����A�*'

learning_rate_1�G�:

loss_1٠u@T�D5       ��]�	�ц����A�*'

learning_rate_1�G�:

loss_1@yq@<�PT5       ��]�	hI�����A�*'

learning_rate_1�G�:

loss_1�g@���5       ��]�	ڳ�����A�*'

learning_rate_1�G�:

loss_1d�l@\Qo�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1]p@��5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�Gt@�G(.5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1tb@HZ�K5       ��]�	�+����A�*'

learning_rate_1�G�:

loss_1�z@��	5       ��]�	[#����A�*'

learning_rate_1�G�:

loss_1�U{@d��5       ��]�	؉9����A�*'

learning_rate_1�G�:

loss_1�rx@����5       ��]�	#7P����A�*'

learning_rate_1�G�:

loss_1�qr@A[��5       ��]�	&kf����A�*'

learning_rate_1�G�:

loss_1��h@�6�5       ��]�	U}����A�*'

learning_rate_1�G�:

loss_1�u@�g�65       ��]�	����A�*'

learning_rate_1�G�:

loss_1�l@��=c5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��h@ɍ�5       ��]�	L�����A�*'

learning_rate_1�G�:

loss_1E>Z@5��5       ��]�	x������A�*'

learning_rate_1�G�:

loss_1�{_@�-��5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�o@H��^5       ��]�	]�����A�*'

learning_rate_1�G�:

loss_1��q@�<0?5       ��]�	�z����A�*'

learning_rate_1�G�:

loss_1U��@	P�5       ��]�	r�0����A�*'

learning_rate_1�G�:

loss_1�j@!"P�5       ��]�	PG����A�*'

learning_rate_1�G�:

loss_1���@���5       ��]�	U�]����A�*'

learning_rate_1�G�:

loss_1/Dp@�5       ��]�	��s����A�*'

learning_rate_1�G�:

loss_1f�n@���5       ��]�	�\�����A�*'

learning_rate_1�G�:

loss_17Hw@!�Q�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1Ϊw@�@� 5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�l@��H5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�Jw@����5       ��]�	�h�����A�*'

learning_rate_1�G�:

loss_1�[@ԇ�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1��c@���5       ��]�	������A�*'

learning_rate_1�G�:

loss_1>�^@��Z5       ��]�	Dg)����A�*'

learning_rate_1�G�:

loss_1-n@u�l5       ��]�	��?����A�*'

learning_rate_1�G�:

loss_1�8r@��5       ��]�	��S����A�*'

learning_rate_1�G�:

loss_1���@3X�5       ��]�	Ԕk����A�*'

learning_rate_1�G�:

loss_10e@��8�5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1��v@|F��5       ��]�	`�����A�*'

learning_rate_1�G�:

loss_179h@���h5       ��]�	b������A�*'

learning_rate_1�G�:

loss_1�q@E�>5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�|�@����5       ��]�	kK�����A�*'

learning_rate_1�G�:

loss_1&�k@�tn5       ��]�	������A�*'

learning_rate_1�G�:

loss_17	�@�� J5       ��]�	c�����A�*'

learning_rate_1�G�:

loss_1m�~@���L5       ��]�	�4����A�*'

learning_rate_1�G�:

loss_1{s@��p5       ��]�	b4����A�*'

learning_rate_1�G�:

loss_1��d@_[��5       ��]�	��J����A�*'

learning_rate_1�G�:

loss_1̻z@���5       ��]�	��`����A�*'

learning_rate_1�G�:

loss_1:z@&��%5       ��]�	b)w����A�*'

learning_rate_1�G�:

loss_1\3u@/��5       ��]�	�T�����A�*'

learning_rate_1�G�:

loss_1�.i@���5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1+In@��.5       ��]�	%�����A�*'

learning_rate_1�G�:

loss_1C�b@ȣN�5       ��]�	�<�����A�*'

learning_rate_1�G�:

loss_1�n@��}5       ��]�	�}�����A�*'

learning_rate_1�G�:

loss_17��@T4�B5       ��]�	8������A�*'

learning_rate_1�G�:

loss_1�v@ӻv'5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1I�o@��,5       ��]�	�8)����A�*'

learning_rate_1�G�:

loss_1�f@t�8�5       ��]�	pN@����A�*'

learning_rate_1�G�:

loss_1ѓe@��^5       ��]�	��V����A�*'

learning_rate_1�G�:

loss_1k�f@o��v5       ��]�	a�m����A�*'

learning_rate_1�G�:

loss_1=!g@A�>>5       ��]�	W�����A�*'

learning_rate_1�G�:

loss_1��f@+y�5       ��]�	�H�����A�*'

learning_rate_1�G�:

loss_1	q@��1�5       ��]�	M������A�*'

learning_rate_1�G�:

loss_1�"b@]A�5       ��]�	�p�����A�*'

learning_rate_1�G�:

loss_1^�i@���5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1��y@�xl*5       ��]�	�b�����A�*'

learning_rate_1�G�:

loss_1.�j@O���5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��l@AU 5       ��]�	="����A�*'

learning_rate_1�G�:

loss_1]bu@���J5       ��]�	�X8����A�*'

learning_rate_1�G�:

loss_1�$q@���%5       ��]�	?�N����A�*'

learning_rate_1�G�:

loss_1�r@��dD5       ��]�	�.e����A�*'

learning_rate_1�G�:

loss_1�ku@���5       ��]�	Z�{����A�*'

learning_rate_1�G�:

loss_1
cp@ّh�5       ��]�	�ב����A�*'

learning_rate_1�G�:

loss_1o`h@^�DO5       ��]�	d=�����A�*'

learning_rate_1�G�:

loss_1Sn@��L�5       ��]�	֒�����A�*'

learning_rate_1�G�:

loss_1QJr@�0�$5       ��]�	d������A�*'

learning_rate_1�G�:

loss_1�oh@�;�Y5       ��]�	Y������A�*'

learning_rate_1�G�:

loss_1�4b@N2��5       ��]�	U����A�*'

learning_rate_1�G�:

loss_1��Z@�[��5       ��]�	(����A�*'

learning_rate_1�G�:

loss_1I�u@���5       ��]�	~�.����A�*'

learning_rate_1�G�:

loss_1��t@�)�k5       ��]�	�E����A�*'

learning_rate_1�G�:

loss_1w�k@�h��5       ��]�	'I[����A�*'

learning_rate_1�G�:

loss_1�'f@֞_	5       ��]�	��q����A�*'

learning_rate_1�G�:

loss_1vAd@οd�5       ��]�	� �����A�*'

learning_rate_1�G�:

loss_1�2h@�%5c5       ��]�	�p�����A�*'

learning_rate_1�G�:

loss_1� r@µ��5       ��]�	O2�����A�*'

learning_rate_1�G�:

loss_1;C|@��c�5       ��]�	g������A�*'

learning_rate_1�G�:

loss_1��i@�aR�5       ��]�	V������A�*'

learning_rate_1�G�:

loss_1�[w@Ӻ15       ��]�	-������A�*'

learning_rate_1�G�:

loss_1��U@CH5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�kw@�G5       ��]�	�8&����A�*'

learning_rate_1�G�:

loss_1a�a@�r��5       ��]�	�<����A�*'

learning_rate_1�G�:

loss_1��d@�b�"5       ��]�	��T����A�*'

learning_rate_1�G�:

loss_1�\@�5       ��]�	�6k����A�*'

learning_rate_1�G�:

loss_1�8v@.\o5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1�df@A$�K5       ��]�	�ǘ����A�*'

learning_rate_1�G�:

loss_18|i@eG�5       ��]�	>������A�*'

learning_rate_1�G�:

loss_1 �p@C>�L5       ��]�	*6�����A�*'

learning_rate_1�G�:

loss_1*;t@�2�N5       ��]�	�w�����A�*'

learning_rate_1�G�:

loss_1i�Z@�\��5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�d@|@6R5       ��]�	ގ
����A�*'

learning_rate_1�G�:

loss_1�SY@��p5       ��]�	-� ����A�*'

learning_rate_1�G�:

loss_1�0y@�f�5       ��]�	E=7����A�*'

learning_rate_1�G�:

loss_1�\@�W��5       ��]�	@�M����A�*'

learning_rate_1�G�:

loss_1��t@���/5       ��]�	�Wf����A�*'

learning_rate_1�G�:

loss_1�R@��t]5       ��]�	a�|����A�*'

learning_rate_1�G�:

loss_1h�l@*���5       ��]�	3�����A�*'

learning_rate_1�G�:

loss_1�s@[���5       ��]�	�J�����A�*'

learning_rate_1�G�:

loss_1h�q@��5       ��]�	������A�*'

learning_rate_1�G�:

loss_1q�`@��5       ��]�	�}�����A�*'

learning_rate_1�G�:

loss_1*f@�~�[5       ��]�	2"�����A�*'

learning_rate_1�G�:

loss_10>i@�gԷ5       ��]�	<.����A�*'

learning_rate_1�G�:

loss_1`�Z@ ��]5       ��]�	ߒ����A�*'

learning_rate_1�G�:

loss_14Vh@F�Z�5       ��]�	A�2����A�*'

learning_rate_1�G�:

loss_1Z�n@�
�5       ��]�	�F����A�*'

learning_rate_1�G�:

loss_1�R}@`��r5       ��]�	�l]����A�*'

learning_rate_1�G�:

loss_1�_@o<�5       ��]�	t����A�*'

learning_rate_1�G�:

loss_1S�l@v��	5       ��]�	������A�*'

learning_rate_1�G�:

loss_1s(c@`�Ƹ5       ��]�	�y�����A�*'

learning_rate_1�G�:

loss_1�[h@t޾�5       ��]�	Y������A�*'

learning_rate_1�G�:

loss_10�x@7�(�5       ��]�	�>�����A�*'

learning_rate_1�G�:

loss_1�u@@N5       ��]�	M������A�*'

learning_rate_1�G�:

loss_1�l@��='5       ��]�	�)�����A�*'

learning_rate_1�G�:

loss_1��i@d�z�5       ��]�	2N����A�*'

learning_rate_1�G�:

loss_1�uo@PP��5       ��]�	��)����A�*'

learning_rate_1�G�:

loss_1U�j@�b�5       ��]�	��?����A�*'

learning_rate_1�G�:

loss_1!�^@#�E5       ��]�	8�V����A�*'

learning_rate_1�G�:

loss_1��j@K;�5       ��]�	2�l����A�*'

learning_rate_1�G�:

loss_1�k@2�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�e_@��ǫ5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�"x@� 3g5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�<k@$��5       ��]�	�^�����A�*'

learning_rate_1�G�:

loss_1�Pf@��H5       ��]�	5������A�*'

learning_rate_1�G�:

loss_1�[�@�*˫5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�c@a�@5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1��Z@���5       ��]�	�C"����A�*'

learning_rate_1�G�:

loss_1�*u@ZS��5       ��]�	Կ8����A�*'

learning_rate_1�G�:

loss_1d�p@�xB�5       ��]�	�O����A�*'

learning_rate_1�G�:

loss_1*hl@H1%v5       ��]�	�Ne����A�*'

learning_rate_1�G�:

loss_1�sn@���,5       ��]�	^�{����A�*'

learning_rate_1�G�:

loss_1�5l@o��5       ��]�	W������A�*'

learning_rate_1�G�:

loss_1Ju@��_�5       ��]�	�"�����A�*'

learning_rate_1�G�:

loss_1�4c@N5�5       ��]�	K�����A�*'

learning_rate_1�G�:

loss_1��i@���5       ��]�	Y �����A�*'

learning_rate_1�G�:

loss_1!w@Y�+5       ��]�	�`�����A�*'

learning_rate_1�G�:

loss_1�%i@o��5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1��[@S��s5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1j�d@~��5       ��]�	0����A�*'

learning_rate_1�G�:

loss_1�"e@KN[<5       ��]�	_xF����A�*'

learning_rate_1�G�:

loss_1��o@�p625       ��]�	T�\����A�*'

learning_rate_1�G�:

loss_1u@> �=5       ��]�	Xs����A�*'

learning_rate_1�G�:

loss_1
�~@� �[5       ��]�	b^�����A�*'

learning_rate_1�G�:

loss_1�Ue@���E5       ��]�	8_�����A�*'

learning_rate_1�G�:

loss_1�)p@�V�5       ��]�	a������A�*'

learning_rate_1�G�:

loss_1~Qk@�,�t5       ��]�	3������A�*'

learning_rate_1�G�:

loss_1oj@^���5       ��]�	������A�*'

learning_rate_1�G�:

loss_1&j@����5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�Wl@�G��5       ��]�	-����A�*'

learning_rate_1�G�:

loss_17s@�l%�5       ��]�	5'����A�*'

learning_rate_1�G�:

loss_1�.h@3i&�5       ��]�	�=����A�*'

learning_rate_1�G�:

loss_1�lg@#;55       ��]�	B|Q����A�*'

learning_rate_1�G�:

loss_1]�v@�A�5       ��]�	��g����A�*'

learning_rate_1�G�:

loss_1+�]@n��5       ��]�	�~����A�*'

learning_rate_1�G�:

loss_1k5o@Y @�5       ��]�	C3�����A�*'

learning_rate_1�G�:

loss_1� d@���5       ��]�	0m�����A�*'

learning_rate_1�G�:

loss_1L�n@��.=5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�5h@H�"5       ��]�	 )�����A�*'

learning_rate_1�G�:

loss_1�kg@���s5       ��]�	�h�����A�*'

learning_rate_1�G�:

loss_1�^@?ĩ�5       ��]�	O����A�*'

learning_rate_1�G�:

loss_1_�a@���5       ��]�	�N����A�*'

learning_rate_1�G�:

loss_1��f@*��5       ��]�	��1����A�*'

learning_rate_1�G�:

loss_1*�m@�a�5       ��]�	Z?H����A�*'

learning_rate_1�G�:

loss_1Nn@��95       ��]�	�^����A�*'

learning_rate_1�G�:

loss_1��a@11h�5       ��]�	��t����A�*'

learning_rate_1�G�:

loss_1��i@��n5       ��]�	
A�����A�*'

learning_rate_1�G�:

loss_1�d@�5�5       ��]�	<�����A�*'

learning_rate_1�G�:

loss_1��X@�Z^D5       ��]�	z{�����A�*'

learning_rate_1�G�:

loss_1.r@6��5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�cj@g��5       ��]�	I�����A�*'

learning_rate_1�G�:

loss_1k@�Mp�5       ��]�	�Q�����A�*'

learning_rate_1�G�:

loss_1d�f@�)E65       ��]�	?����A�*'

learning_rate_1�G�:

loss_16ja@�w5       ��]�	��)����A�*'

learning_rate_1�G�:

loss_1}Fi@��P�5       ��]�	��?����A�*'

learning_rate_1�G�:

loss_1��h@���5       ��]�	KV����A�*'

learning_rate_1�G�:

loss_1�m@����5       ��]�	pl����A�*'

learning_rate_1�G�:

loss_1ta@��]5       ��]�	�H�����A�*'

learning_rate_1�G�:

loss_1�z@�E�5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1P�h@��A�5       ��]�	�<�����A�*'

learning_rate_1�G�:

loss_1�W@Wi�5       ��]�	R�����A�*'

learning_rate_1�G�:

loss_1�'R@�g5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�v@�w�5       ��]�	wV�����A�*'

learning_rate_1�G�:

loss_1��r@��u�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��g@"<l�5       ��]�	�����A�*'

learning_rate_1�G�:

loss_1_d@
\�:5       ��]�	m�5����A�*'

learning_rate_1�G�:

loss_1��k@Ͽ�5       ��]�	�
L����A�*'

learning_rate_1�G�:

loss_1�sx@G�825       ��]�	epb����A�*'

learning_rate_1�G�:

loss_1�e@}��5       ��]�	m�x����A�*'

learning_rate_1�G�:

loss_19�_@	��5       ��]�	C������A�*'

learning_rate_1�G�:

loss_1�9m@7�d5       ��]�	�^�����A�*'

learning_rate_1�G�:

loss_1�_@u��5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1&/r@��P5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1��p@��L�5       ��]�	�2�����A�*'

learning_rate_1�G�:

loss_1��w@`g�5       ��]�	յ�����A�*'

learning_rate_1�G�:

loss_1�@{@�qE5       ��]�	������A�*'

learning_rate_1�G�:

loss_1]hQ@�a[+5       ��]�	 -����A�*'

learning_rate_1�G�:

loss_1��f@"ٷ�5       ��]�	[SC����A�*'

learning_rate_1�G�:

loss_1y�u@a*"5       ��]�	1�Y����A�*'

learning_rate_1�G�:

loss_1�k@��[�5       ��]�	1p����A�*'

learning_rate_1�G�:

loss_1?�W@��<m5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1y�z@'$j�5       ��]�	=ݜ����A�*'

learning_rate_1�G�:

loss_1hk@|�N5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��k@a�Z5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1f�j@����5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�Pm@�uO�5       ��]�	4e�����A�*'

learning_rate_1�G�:

loss_1�4[@y�v35       ��]�	�u����A�*'

learning_rate_1�G�:

loss_1E�m@�}95       ��]�	��"����A�*'

learning_rate_1�G�:

loss_1�p@�eq5       ��]�	�u9����A�*'

learning_rate_1�G�:

loss_1�\u@����5       ��]�	bO����A�*'

learning_rate_1�G�:

loss_1��V@��`�5       ��]�	�8e����A�*'

learning_rate_1�G�:

loss_1�@� ��5       ��]�	�{����A�*'

learning_rate_1�G�:

loss_1��_@Z�y5       ��]�	������A�*'

learning_rate_1�G�:

loss_1ʮg@e�'�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1J�y@�9�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1��k@.	[�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�jl@4:�\5       ��]�	U������A�*'

learning_rate_1�G�:

loss_1��j@�F	i5       ��]�	X������A�*'

learning_rate_1�G�:

loss_1��]@����5       ��]�	b�����A�*'

learning_rate_1�G�:

loss_1Nf@��b5       ��]�	g�*����A�*'

learning_rate_1�G�:

loss_1X�d@	v��5       ��]�	\�A����A�*'

learning_rate_1�G�:

loss_1%�b@ݟH5       ��]�	��W����A�*'

learning_rate_1�G�:

loss_1~h@hC!�5       ��]�	��m����A�*'

learning_rate_1�G�:

loss_1vYb@I�#p5       ��]�	b�����A�*'

learning_rate_1�G�:

loss_1�Mm@�M#5       ��]�	������A�*'

learning_rate_1�G�:

loss_1�e@�&�5       ��]�	�A�����A�*'

learning_rate_1�G�:

loss_1��V@��UZ5       ��]�	g������A�*'

learning_rate_1�G�:

loss_1d�p@J25�5       ��]�	t������A�*'

learning_rate_1�G�:

loss_1�3w@/(;5       ��]�	�U�����A�*'

learning_rate_1�G�:

loss_1u�\@�A�5       ��]�	�
����A�*'

learning_rate_1�G�:

loss_1��_@����5       ��]�	�+!����A�*'

learning_rate_1�G�:

loss_1td@�m�45       ��]�	J5����A�*'

learning_rate_1�G�:

loss_1N�~@���5       ��]�	�:K����A�*'

learning_rate_1�G�:

loss_1r�f@,���5       ��]�	gZa����A�*'

learning_rate_1�G�:

loss_1��]@�a��5       ��]�	��x����A�*'

learning_rate_1�G�:

loss_1J9[@��5       ��]�	I�����A�*'

learning_rate_1�G�:

loss_1��W@ʀ"5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1mg@�1��5       ��]�	������A�*'

learning_rate_1�G�:

loss_1+�f@�٢�5       ��]�	������A�*'

learning_rate_1�G�:

loss_1��a@A�5       ��]�	�������A�*'

learning_rate_1�G�:

loss_1�|@��5       ��]�	&����A�*'

learning_rate_1�G�:

loss_1ww@��K�5       ��]�	p�����A�*'

learning_rate_1�G�:

loss_14yh@�M��5       ��]�	�-����A�*'

learning_rate_1�G�:

loss_1�m@&C�A5       ��]�	��D����A�*'

learning_rate_1�G�:

loss_1��a@�(�5       ��]�	~xZ����A�*'

learning_rate_1�G�:

loss_1yBv@x���5       ��]�	��p����A�*'

learning_rate_1�G�:

loss_1t�O@-���5       ��]�	�*�����A�*'

learning_rate_1�G�:

loss_1-�c@�j?�5       ��]�	U������A�*'

learning_rate_1�G�:

loss_1�[@�ҝ5       ��]�	*�����A�*'

learning_rate_1�G�:

loss_177p@�C�h5       ��]�	F�����A�*'

learning_rate_1�G�:

loss_1��d@�S,b5       ��]�	�������A�*'

learning_rate_1�G�:

loss_17�h@㘲�5       ��]�	O������A�*'

learning_rate_1�G�:

loss_1��p@ɠ]15       ��]�	&E����A�*'

learning_rate_1�G�:

loss_1�^@��;�5       ��]�	
y#����A�*'

learning_rate_1�G�:

loss_1�[o@��B 5       ��]�	O�9����A�*'

learning_rate_1�G�:

loss_1�me@J�)5       ��]�	yUP����A�*'

learning_rate_1�G�:

loss_1�d@m*�5       ��]�	�g����A�*'

learning_rate_1�G�:

loss_1p�c@��5       ��]�	�Y����A�*'

learning_rate_1�G�:

loss_1�gY@��5       ��]�	Y������A�*'

learning_rate_1���:

loss_1��k@��m5       ��]�	� �����A�*'

learning_rate_1���:

loss_1�e@���a5       ��]�	�;�����A�*'

learning_rate_1���:

loss_1!d@\!�,5       ��]�	ח�����A�*'

learning_rate_1���:

loss_1݅d@ݲ�5       ��]�	�k�����A�*'

learning_rate_1���:

loss_1��n@1� z5       ��]�	X�����A�*'

learning_rate_1���:

loss_1{S@��v�5       ��]�	F#����A�*'

learning_rate_1���:

loss_17�d@�O5       ��]�	Ю:����A�*'

learning_rate_1���:

loss_1�}_@��i�5       ��]�	��P����A�*'

learning_rate_1���:

loss_1�`@&��}5       ��]�	�0g����A�*'

learning_rate_1���:

loss_1��i@�̈́5       ��]�	J}����A�*'

learning_rate_1���:

loss_1��k@� �]5       ��]�	�{�����A�*'

learning_rate_1���:

loss_1�b@��5       ��]�	������A�*'

learning_rate_1���:

loss_1B�V@Eu&5       ��]�	T\�����A�*'

learning_rate_1���:

loss_11c@+:�5       ��]�	|������A�*'

learning_rate_1���:

loss_1r�u@U��5       ��]�	�������A�*'

learning_rate_1���:

loss_1�tj@(�<n5       ��]�	�����A�*'

learning_rate_1���:

loss_1�ft@�)Z�5       ��]�	uW����A�*'

learning_rate_1���:

loss_1�Y@���5       ��]�	��3����A�*'

learning_rate_1���:

loss_1�	m@围b5       ��]�	�AK����A�*'

learning_rate_1���:

loss_1@R@a�{5       ��]�	oa����A�*'

learning_rate_1���:

loss_1�)f@ ,��5       ��]�	�x����A�*'

learning_rate_1���:

loss_1�`@��g�5       ��]�	*������A�*'

learning_rate_1���:

loss_1��]@_�5       ��]�	�����A�*'

learning_rate_1���:

loss_1Kf@ܣ�5       ��]�	�������A�*'

learning_rate_1���:

loss_1r5l@���5       ��]�	�2�����A�*'

learning_rate_1���:

loss_1"�Y@�>ȟ5       ��]�	g�����A�*'

learning_rate_1���:

loss_1�qX@_��5       ��]�	~������A�*'

learning_rate_1���:

loss_1ݤ_@	!�:5       ��]�	y�����A�*'

learning_rate_1���:

loss_1��Y@�Ē�5       ��]�	�+����A�*'

learning_rate_1���:

loss_1�(j@�~85       ��]�	�GA����A�*'

learning_rate_1���:

loss_1fR@�?��5       ��]�	�W����A�*'

learning_rate_1���:

loss_1]�^@8�X5       ��]�	�Mm����A�*'

learning_rate_1���:

loss_1�\@5       ��]�	~�����A�*'

learning_rate_1���:

loss_1�\k@���95       ��]�	������A�*'

learning_rate_1���:

loss_1c�f@� n�5       ��]�	'ׯ����A�*'

learning_rate_1���:

loss_1��a@�7��5       ��]�	=�����A�*'

learning_rate_1���:

loss_1vPg@>&�<5       ��]�	�������A�*'

learning_rate_1���:

loss_1Ng@�=�o5       ��]�	i������A�*'

learning_rate_1���:

loss_1�dc@�MԊ5       ��]�	��	����A�*'

learning_rate_1���:

loss_1�_@y���5       ��]�	; ����A�*'

learning_rate_1���:

loss_1�Q@b�z�5       ��]�	&�7����A�*'

learning_rate_1���:

loss_1�ze@�y�5       ��]�	�N����A�*'

learning_rate_1���:

loss_1�_c@ac�S5       ��]�	��e����A�*'

learning_rate_1���:

loss_1��l@);�T5       ��]�	Ʊ{����A�*'

learning_rate_1���:

loss_1�zh@��S.5       ��]�	�:�����A�*'

learning_rate_1���:

loss_1��V@Af85       ��]�	?j�����A�*'

learning_rate_1���:

loss_1�Sq@PM�5       ��]�	������A�*'

learning_rate_1���:

loss_1�a@��M�5       ��]�	�D�����A�*'

learning_rate_1���:

loss_1թW@��5       ��]�	������A�*'

learning_rate_1���:

loss_1_�F@9�F15       ��]�	������A�*'

learning_rate_1���:

loss_16�d@�O�5       ��]�	�����A�*'

learning_rate_1���:

loss_1�Zc@���5       ��]�	?�2����A�*'

learning_rate_1���:

loss_1""a@�;�5       ��]�	f?J����A�*'

learning_rate_1���:

loss_1g6e@�E��5       ��]�	
�`����A�*'

learning_rate_1���:

loss_1�u@�()5       ��]�	˘v����A�*'

learning_rate_1���:

loss_1ڋ^@�"�5       ��]�	�������A�*'

learning_rate_1���:

loss_1-YU@Ǣ�A5       ��]�	�������A�*'

learning_rate_1���:

loss_1�c@T̋Z5       ��]�	|@�����A�*'

learning_rate_1���:

loss_1U�k@YQ45       ��]�	�������A�*'

learning_rate_1���:

loss_1�c@��x�5       ��]�	�������A�*'

learning_rate_1���:

loss_1�e@&�	#5       ��]�	�������A�*'

learning_rate_1���:

loss_1ufG@xOxW5       ��]�	�����A�*'

learning_rate_1���:

loss_1��b@���5       ��]�	�+����A�*'

learning_rate_1���:

loss_1'�b@�	�H5       ��]�	�[A����A�*'

learning_rate_1���:

loss_1=�\@J��I5       ��]�	N�W����A�*'

learning_rate_1���:

loss_1�qe@Wҥ�5       ��]�	��m����A�*'

learning_rate_1���:

loss_1G�Z@FAzX5       ��]�	]�����A�*'

learning_rate_1���:

loss_17�_@i��`5       ��]�	BF�����A�*'

learning_rate_1���:

loss_1� c@$�U�5       ��]�	P~�����A�*'

learning_rate_1���:

loss_1��Z@�'i�5       ��]�	������A�*'

learning_rate_1���:

loss_1��]@�Zϣ5       ��]�	*�����A�*'

learning_rate_1���:

loss_1!;]@YG�5       ��]�	Q������A�*'

learning_rate_1���:

loss_1X�h@���<5       ��]�	�O
����A�*'

learning_rate_1���:

loss_1VAe@��:5       ��]�	�{ ����A�*'

learning_rate_1���:

loss_1� _@4���5       ��]�	�6����A�*'

learning_rate_1���:

loss_1}�j@��W5       ��]�	�L����A�*'

learning_rate_1���:

loss_1V�Z@O��5       ��]�	W�b����A�*'

learning_rate_1���:

loss_1�W@�i�5       ��]�	^y����A�*'

learning_rate_1���:

loss_1h�m@|��5       ��]�	�֏����A�*'

learning_rate_1���:

loss_1	�b@�kg5       ��]�	k�����A�*'

learning_rate_1���:

loss_1J�i@?fWb5       ��]�	������A�*'

learning_rate_1���:

loss_1�\b@h&5       ��]�	�������A�*'

learning_rate_1���:

loss_1B�b@[��.5       ��]�	�������A�*'

learning_rate_1���:

loss_1>5f@���u5       ��]�	D� ����A�*'

learning_rate_1���:

loss_1G�a@�"�5       ��]�	g�����A�*'

learning_rate_1���:

loss_1{U@�^a%5       ��]�	w�,����A�*'

learning_rate_1���:

loss_1U]a@`�5       ��]�	��C����A�*'

learning_rate_1���:

loss_1X;d@�Q�5       ��]�	<X[����A�*'

learning_rate_1���:

loss_1�Q`@w�5       ��]�	�sq����A�*'

learning_rate_1���:

loss_1�4Y@��%�5       ��]�	·����A�*'

learning_rate_1���:

loss_1��b@k_%�5       ��]�	�������A�*'

learning_rate_1���:

loss_12�j@-��5       ��]�	q%�����A�*'

learning_rate_1���:

loss_1Z@�*��5       ��]�	Rm�����A�*'

learning_rate_1���:

loss_1�)_@��G5       ��]�	������A�*'

learning_rate_1���:

loss_1d?h@(Ŧ�5       ��]�	EQ�����A�*'

learning_rate_1���:

loss_1��Y@�"�q5       ��]�	�����A�*'

learning_rate_1���:

loss_1��h@�F��5       ��]�	%����A�*'

learning_rate_1���:

loss_1Y�g@��'5       ��]�	�E;����A�*'

learning_rate_1���:

loss_1Z@[3s5       ��]�	�pQ����A�*'

learning_rate_1���:

loss_1�Fc@��j�5       ��]�	#�g����A�*'

learning_rate_1���:

loss_1��Y@�.�5       ��]�	p�}����A�*'

learning_rate_1���:

loss_1�Cd@�u��5       ��]�	������A�*'

learning_rate_1���:

loss_1ŉ`@][�F5       ��]�	H�����A�*'

learning_rate_1���:

loss_1	(s@`��5       ��]�	m�����A�*'

learning_rate_1���:

loss_1��c@/x�L5       ��]�	������A�*'

learning_rate_1���:

loss_1-�e@��4�5       ��]�	{A�����A�*'

learning_rate_1���:

loss_1>�z@ʺkQ5       ��]�	�%����A�*'

learning_rate_1���:

loss_1´m@��I5       ��]�	Ol����A�*'

learning_rate_1���:

loss_1�.]@�wR5       ��]�	# 2����A�*'

learning_rate_1���:

loss_1�ta@��7�5       ��]�	�CH����A�*'

learning_rate_1���:

loss_1&�`@�t.�5       ��]�	s�_����A�*'

learning_rate_1���:

loss_1!�X@�~5       ��]�	pfv����A�*'

learning_rate_1���:

loss_1v�`@o�95       ��]�	렌����A�*'

learning_rate_1���:

loss_1�>f@>Y5       ��]�	U0�����A�*'

learning_rate_1���:

loss_1� n@i���5       ��]�	�ù����A�*'

learning_rate_1���:

loss_1ڷ`@ࡓ�5       ��]�	�,�����A�*'

learning_rate_1���:

loss_1i`@%��z5       ��]�	�^�����A�*'

learning_rate_1���:

loss_1?�X@���x5       ��]�	�������A�*'

learning_rate_1���:

loss_1�p@��/�5       ��]�	�����A�*'

learning_rate_1���:

loss_1��]@rJ�c5       ��]�	�]*����A�*'

learning_rate_1���:

loss_1��a@�A��5       ��]�	�A@����A�*'

learning_rate_1���:

loss_1�a@ߤ}�5       ��]�	�V����A�*'

learning_rate_1���:

loss_1P�e@[�v�5       ��]�	��l����A�*'

learning_rate_1���:

loss_1�k@�Z��5       ��]�	������A�*'

learning_rate_1���:

loss_1��s@�1��5       ��]�	G�����A�*'

learning_rate_1���:

loss_1�b[@WP�5       ��]�	l.�����A�*'

learning_rate_1���:

loss_1��d@���5       ��]�	�p�����A�*'

learning_rate_1���:

loss_1� g@�]T5       ��]�	e������A�*'

learning_rate_1���:

loss_1*�b@r$�5       ��]�	xv�����A�*'

learning_rate_1���:

loss_1��a@(�5       ��]�	ȕ����A�*'

learning_rate_1���:

loss_1Ld@�g��5       ��]�	o�����A�*'

learning_rate_1���:

loss_1�vL@�Dj5       ��]�	V6����A�*'

learning_rate_1���:

loss_1�]@Ha7�5       ��]�	��L����A�*'

learning_rate_1���:

loss_1�n@�w�|5       ��]�	-c����A�*'

learning_rate_1���:

loss_1��r@���>5       ��]�	�vy����A�*'

learning_rate_1���:

loss_1��_@W�5       ��]�	�������A�*'

learning_rate_1���:

loss_1yji@�\��5       ��]�	ߥ����A�*'

learning_rate_1���:

loss_1�\@)bF�5       ��]�	������A�*'

learning_rate_1���:

loss_17mi@1���5       ��]�	!\�����A�*'

learning_rate_1���:

loss_1�
`@,k��5       ��]�	�������A�*'

learning_rate_1���:

loss_1��Z@����5       ��]�	�f�����A�*'

learning_rate_1���:

loss_1��d@��5       ��]�	������A�*'

learning_rate_1���:

loss_1�`@���(5       ��]�	(,����A�*'

learning_rate_1���:

loss_1��g@�'��5       ��]�	�WB����A�*'

learning_rate_1���:

loss_1~�b@�F��5       ��]�	��X����A�*'

learning_rate_1���:

loss_1+Vg@�5       ��]�	��n����A�*'

learning_rate_1���:

loss_1�^u@�@5       ��]�	(�����A�*'

learning_rate_1���:

loss_1�_c@�AS�5       ��]�	�ڛ����A�*'

learning_rate_1���:

loss_1)Y@��T5       ��]�	�)�����A�*'

learning_rate_1���:

loss_1&�g@OA�5       ��]�	au�����A�*'

learning_rate_1���:

loss_1��e@��G�5       ��]�	z������A�*'

learning_rate_1���:

loss_1��c@��.�5       ��]�	�G�����A�	*'

learning_rate_1���:

loss_1��c@��̌5       ��]�	�z����A�	*'

learning_rate_1���:

loss_1ٳ[@{V��5       ��]�	�!����A�	*'

learning_rate_1���:

loss_1�l@C:T�5       ��]�	G8����A�	*'

learning_rate_1���:

loss_1�HS@��5       ��]�	�%N����A�	*'

learning_rate_1���:

loss_1��T@���5       ��]�	��d����A�	*'

learning_rate_1���:

loss_1�Y@g���5       ��]�	/{����A�	*'

learning_rate_1���:

loss_1�R@���:5       ��]�	bu�����A�	*'

learning_rate_1���:

loss_1LWa@�5       ��]�	H������A�	*'

learning_rate_1���:

loss_1~l@���5       ��]�	? �����A�	*'

learning_rate_1���:

loss_1�V]@����5       ��]�	�������A�	*'

learning_rate_1���:

loss_1-Hg@�S�5       ��]�	M<�����A�	*'

learning_rate_1���:

loss_1��e@"�+�5       ��]�	|�����A�	*'

learning_rate_1���:

loss_1�\@���z5       ��]�	������A�	*'

learning_rate_1���:

loss_1v�X@ila5       ��]�	c�.����A�	*'

learning_rate_1���:

loss_1Z]f@ԛ%�5       ��]�	��D����A�	*'

learning_rate_1���:

loss_17?d@C�Z�5       ��]�	�([����A�	*'

learning_rate_1���:

loss_1tfQ@�\��5       ��]�	�|q����A�	*'

learning_rate_1���:

loss_14#b@b�`5       ��]�	X�����A�	*'

learning_rate_1���:

loss_1�d@f�~z5       ��]�	8������A�	*'

learning_rate_1���:

loss_1�P@��>�5       ��]�	ͷ�����A�	*'

learning_rate_1���:

loss_1�W@#p�5       ��]�	�������A�	*'

learning_rate_1���:

loss_1~�[@�'�5       ��]�	������A�	*'

learning_rate_1���:

loss_1��N@y �5       ��]�	�q�����A�	*'

learning_rate_1���:

loss_1�%L@Szs5       ��]�	L ����A�	*'

learning_rate_1���:

loss_1�"^@���5       ��]�	�)����A�	*'

learning_rate_1���:

loss_1�Ia@��0�5       ��]�	�y@����A�	*'

learning_rate_1���:

loss_1_n@��5       ��]�	��V����A�	*'

learning_rate_1���:

loss_1�a@�Z5       ��]�	Еn����A�	*'

learning_rate_1���:

loss_1B9Q@
�U5       ��]�	�τ����A�	*'

learning_rate_1���:

loss_1wAN@�� k5       ��]�	c�����A�	*'

learning_rate_1���:

loss_1y�T@���5       ��]�	i)�����A�	*'

learning_rate_1���:

loss_1�_@��4�5       ��]�	=������A�	*'

learning_rate_1���:

loss_18�^@r�45       ��]�	�0�����A�	*'

learning_rate_1���:

loss_1�eQ@l&'5       ��]�	�l�����A�	*'

learning_rate_1���:

loss_17�g@Ϛ5       ��]�	@� ���A�	*'

learning_rate_1���:

loss_1��Y@���5       ��]�	��! ���A�	*'

learning_rate_1���:

loss_1��S@*�;5       ��]�	�@8 ���A�	*'

learning_rate_1���:

loss_1��]@W5�5       ��]�	E�N ���A�	*'

learning_rate_1���:

loss_1�ZV@�:J5       ��]�	��d ���A�	*'

learning_rate_1���:

loss_1��d@���5       ��]�	�{ ���A�	*'

learning_rate_1���:

loss_1��[@��.�5       ��]�	!D� ���A�	*'

learning_rate_1���:

loss_1%"Y@���q5       ��]�	�^� ���A�	*'

learning_rate_1���:

loss_1�`@T��5       ��]�	z�� ���A�	*'

learning_rate_1���:

loss_1@�b@~!��5       ��]�	�i� ���A�	*'

learning_rate_1���:

loss_1�h@��5       ��]�	��� ���A�	*'

learning_rate_1���:

loss_1@�_@�w5       ��]�	����A�	*'

learning_rate_1���:

loss_193T@�d�5       ��]�	F����A�	*'

learning_rate_1���:

loss_1�nd@���g5       ��]�	�/���A�	*'

learning_rate_1���:

loss_12 V@O��o5       ��]�	/F���A�	*'

learning_rate_1���:

loss_1
a@�SY�5       ��]�	,+\���A�	*'

learning_rate_1���:

loss_1��]@�[��5       ��]�	��r���A�	*'

learning_rate_1���:

loss_1�`@7�5       ��]�	�8����A�	*'

learning_rate_1���:

loss_1�U@�y[5       ��]�	�����A�	*'

learning_rate_1���:

loss_1#�Y@-3�i5       ��]�	l�����A�	*'

learning_rate_1���:

loss_1~cg@BLk�5       ��]�	V;����A�	*'

learning_rate_1���:

loss_1�Bm@;,�5       ��]�	JN����A�	*'

learning_rate_1���:

loss_1�e@40�5       ��]�	������A�	*'

learning_rate_1���:

loss_1�Lg@?c5       ��]�	�
���A�	*'

learning_rate_1���:

loss_1su@����5       ��]�	�T(���A�	*'

learning_rate_1���:

loss_1kd@�v�<5       ��]�	�>���A�	*'

learning_rate_1���:

loss_1��p@:>�5       ��]�	��R���A�	*'

learning_rate_1���:

loss_1u1r@�?E35       ��]�	��h���A�	*'

learning_rate_1���:

loss_1��Z@;_��5       ��]�	�����A�	*'

learning_rate_1���:

loss_1�:|@վ��5       ��]�	�1����A�	*'

learning_rate_1���:

loss_1@�_@ߌt5       ��]�	�x����A�	*'

learning_rate_1���:

loss_1�VX@�S-5       ��]�	~i����A�	*'

learning_rate_1���:

loss_1�9g@y��5       ��]�	������A�	*'

learning_rate_1���:

loss_1}�a@��\L5       ��]�	/�����A�	*'

learning_rate_1���:

loss_1pH@�$��5       ��]�	B���A�	*'

learning_rate_1���:

loss_1`�N@� y<5       ��]�	�S���A�	*'

learning_rate_1���:

loss_1�[\@��}5       ��]�	Pc1���A�	*'

learning_rate_1���:

loss_1�$P@�?+�5       ��]�	_TH���A�	*'

learning_rate_1���:

loss_14w]@0$�}5       ��]�	x�^���A�	*'

learning_rate_1���:

loss_1t�]@�(�P5       ��]�	H&u���A�	*'

learning_rate_1���:

loss_1��c@O�:�5       ��]�	�'����A�	*'

learning_rate_1���:

loss_1va@mJ5       ��]�	_����A�	*'

learning_rate_1���:

loss_1�=[@���?5       ��]�	�ķ���A�	*'

learning_rate_1���:

loss_1#q]@�eK5       ��]�	|`����A�	*'

learning_rate_1���:

loss_1� \@%�/�5       ��]�	'�����A�	*'

learning_rate_1���:

loss_1��\@�Nu5       ��]�	������A�	*'

learning_rate_1���:

loss_1�a@�Ī:5       ��]�	Ds���A�	*'

learning_rate_1���:

loss_1��[@�]�5       ��]�	��*���A�	*'

learning_rate_1���:

loss_1V#b@9�?G5       ��]�	�B���A�	*'

learning_rate_1���:

loss_1	a@�O��5       ��]�	�Y���A�	*'

learning_rate_1���:

loss_1S�d@�@.5       ��]�	Po���A�	*'

learning_rate_1���:

loss_1��M@�B��5       ��]�	=օ���A�	*'

learning_rate_1���:

loss_1�!`@���v5       ��]�	 �����A�	*'

learning_rate_1���:

loss_1-�W@9?�w5       ��]�	T�����A�	*'

learning_rate_1���:

loss_1)R@¤n�5       ��]�	�����A�	*'

learning_rate_1���:

loss_1�V_@���5       ��]�	>�����A�	*'

learning_rate_1���:

loss_1��e@�Ž�5       ��]�	������A�	*'

learning_rate_1���:

loss_1gM@�M95       ��]�	�J���A�	*'

learning_rate_1���:

loss_1#f@���5       ��]�	��#���A�	*'

learning_rate_1���:

loss_137W@�g��5       ��]�	e�9���A�	*'

learning_rate_1���:

loss_1oPe@x)5       ��]�	�Q���A�	*'

learning_rate_1���:

loss_1��c@�	0=5       ��]�	&g���A�	*'

learning_rate_1���:

loss_1U�i@��5�5       ��]�	FY}���A�	*'

learning_rate_1���:

loss_1=#X@�A�5       ��]�	6�����A�	*'

learning_rate_1���:

loss_1�S@�m��5       ��]�	�����A�	*'

learning_rate_1���:

loss_1k�j@N�i(5       ��]�	������A�	*'

learning_rate_1���:

loss_1��c@ZAcY5       ��]�	]0����A�	*'

learning_rate_1���:

loss_1�Pf@%k�95       ��]�	�h����A�	*'

learning_rate_1���:

loss_1�^^@�4�5       ��]�	����A�	*'

learning_rate_1���:

loss_1jkZ@�q��5       ��]�	#����A�	*'

learning_rate_1���:

loss_1Id@ ��5       ��]�	�1���A�	*'

learning_rate_1���:

loss_1�Z@��c�5       ��]�	iAG���A�	*'

learning_rate_1���:

loss_1r�`@,V�5       ��]�	+�]���A�	*'

learning_rate_1���:

loss_1֤\@�b�5       ��]�	�>t���A�	*'

learning_rate_1���:

loss_15`@�b�5       ��]�	������A�	*'

learning_rate_1���:

loss_1��k@���H5       ��]�	�נ���A�	*'

learning_rate_1���:

loss_1�Z@[�F�5       ��]�	�����A�	*'

learning_rate_1���:

loss_1�Q@�;ӭ5       ��]�	������A�	*'

learning_rate_1���:

loss_1�tU@l�H�5       ��]�	������A�	*'

learning_rate_1���:

loss_1�H@CQu�5       ��]�	����A�	*'

learning_rate_1���:

loss_1��X@u�� 5       ��]�	�|���A�	*'

learning_rate_1���:

loss_1E�m@��/n5       ��]�	�'���A�	*'

learning_rate_1���:

loss_1�j@��
�5       ��]�	�>���A�	*'

learning_rate_1���:

loss_1��O@L�?�5       ��]�	�0T���A�	*'

learning_rate_1���:

loss_1$oS@
�5       ��]�	�Uj���A�	*'

learning_rate_1���:

loss_1�"l@��D�5       ��]�	L�����A�	*'

learning_rate_1���:

loss_1��W@��=�5       ��]�	������A�	*'

learning_rate_1���:

loss_1�f`@��K5       ��]�	�Y����A�	*'

learning_rate_1���:

loss_1��W@}���5       ��]�	������A�	*'

learning_rate_1���:

loss_1B�_@��@O5       ��]�	������A�	*'

learning_rate_1���:

loss_1*.n@��l�5       ��]�	�W����A�	*'

learning_rate_1���:

loss_1v<K@eT��5       ��]�	�e���A�	*'

learning_rate_1���:

loss_1�S@0�e�5       ��]�	i����A�	*'

learning_rate_1���:

loss_1;Q@5���5       ��]�	��3���A�
*'

learning_rate_1���:

loss_1jV@��O5       ��]�	{(K���A�
*'

learning_rate_1���:

loss_1P)W@�*<�5       ��]�	vxa���A�
*'

learning_rate_1���:

loss_1N�e@05       ��]�	}�w���A�
*'

learning_rate_1���:

loss_1ov]@�˚5       ��]�	�����A�
*'

learning_rate_1���:

loss_1��S@�8H5       ��]�	\����A�
*'

learning_rate_1���:

loss_1LQb@�_ͧ5       ��]�	������A�
*'

learning_rate_1���:

loss_1-t@6�tJ5       ��]�	������A�
*'

learning_rate_1���:

loss_1�S@A��/5       ��]�	q�����A�
*'

learning_rate_1���:

loss_11�R@�� 5       ��]�	.�����A�
*'

learning_rate_1���:

loss_1�D]@��2_5       ��]�	�n	���A�
*'

learning_rate_1���:

loss_1_-T@���5       ��]�	�+	���A�
*'

learning_rate_1���:

loss_1$RS@w��5       ��]�		RA	���A�
*'

learning_rate_1���:

loss_1��`@
Ӄ:5       ��]�	��W	���A�
*'

learning_rate_1���:

loss_1�v^@�dݯ5       ��]�	Zn	���A�
*'

learning_rate_1���:

loss_16?[@��Q5       ��]�	W��	���A�
*'

learning_rate_1���:

loss_1�mf@�j/O5       ��]�	��	���A�
*'

learning_rate_1���:

loss_1�VV@�a�5       ��]�	�	���A�
*'

learning_rate_1���:

loss_1I�\@�룺5       ��]�	?��	���A�
*'

learning_rate_1���:

loss_1��^@�7�5       ��]�	Q��	���A�
*'

learning_rate_1���:

loss_1�dh@5e�	5       ��]�	�N�	���A�
*'

learning_rate_1���:

loss_1\�S@Vu�-5       ��]�	��
���A�
*'

learning_rate_1���:

loss_1-�W@"��5       ��]�	��!
���A�
*'

learning_rate_1���:

loss_1�rh@#�NZ5       ��]�	��7
���A�
*'

learning_rate_1���:

loss_1o�T@Z���5       ��]�	�(N
���A�
*'

learning_rate_1���:

loss_1�[@��5       ��]�	�d
���A�
*'

learning_rate_1���:

loss_1��c@��P5       ��]�	z
���A�
*'

learning_rate_1���:

loss_1f�[@�.�5       ��]�	��
���A�
*'

learning_rate_1���:

loss_1��g@ʹ��5       ��]�	�%�
���A�
*'

learning_rate_1���:

loss_1�h@q�9$5       ��]�	'�
���A�
*'

learning_rate_1���:

loss_1ݚJ@'ᚪ5       ��]�	>��
���A�
*'

learning_rate_1���:

loss_1m�Q@�"��5       ��]�	��
���A�
*'

learning_rate_1���:

loss_1w�W@�<�75       ��]�	� ���A�
*'

learning_rate_1���:

loss_1�tW@��*5       ��]�	j����A�
*'

learning_rate_1���:

loss_1>a@!�m�5       ��]�	{-���A�
*'

learning_rate_1���:

loss_1��]@R�v�5       ��]�	�C���A�
*'

learning_rate_1���:

loss_1݄d@D�.�5       ��]�	GSZ���A�
*'

learning_rate_1���:

loss_1�Fd@�S5       ��]�	�p���A�
*'

learning_rate_1���:

loss_1��N@|m�05       ��]�	M:����A�
*'

learning_rate_1���:

loss_1�[@��'D5       ��]�	+ӝ���A�
*'

learning_rate_1���:

loss_1g^@��5       ��]�	�ʳ���A�
*'

learning_rate_1���:

loss_1��Y@���45       ��]�	�����A�
*'

learning_rate_1���:

loss_1�`@mGZ�5       ��]�	az����A�
*'

learning_rate_1���:

loss_1ÿY@k��5       ��]�	�����A�
*'

learning_rate_1���:

loss_1��[@�h֣5       ��]�	?}���A�
*'

learning_rate_1���:

loss_1p�d@��K�5       ��]�	@�#���A�
*'

learning_rate_1���:

loss_1��f@��bg5       ��]�	��;���A�
*'

learning_rate_1���:

loss_1�oR@k@�5       ��]�	�;R���A�
*'

learning_rate_1���:

loss_1�Cc@�*|5       ��]�	/=f���A�
*'

learning_rate_1���:

loss_1[�c@�c�5       ��]�	�j|���A�
*'

learning_rate_1���:

loss_1�]@��5       ��]�	1�����A�
*'

learning_rate_1���:

loss_1|�M@�D�5       ��]�	�����A�
*'

learning_rate_1���:

loss_1v�_@��85       ��]�	�S����A�
*'

learning_rate_1���:

loss_1_@O�2�5       ��]�	������A�
*'

learning_rate_1���:

loss_1K@����5       ��]�	�.����A�
*'

learning_rate_1���:

loss_1S=f@�`ؠ5       ��]�	Ҝ���A�
*'

learning_rate_1���:

loss_1�]@��6:5       ��]�	�����A�
*'

learning_rate_1���:

loss_1uG@]���5       ��]�	-A0���A�
*'

learning_rate_1���:

loss_1�6c@�>�5       ��]�	�F���A�
*'

learning_rate_1���:

loss_1��\@� :�5       ��]�	��\���A�
*'

learning_rate_1���:

loss_1ՠ[@[��5       ��]�	\�r���A�
*'

learning_rate_1���:

loss_1Z�f@��h�5       ��]�	I4����A�
*'

learning_rate_1���:

loss_1Ǳa@	+س5       ��]�	}<����A�
*'

learning_rate_1���:

loss_1��_@���m5       ��]�	�{����A�
*'

learning_rate_1���:

loss_1��K@�fSw5       ��]�	�����A�
*'

learning_rate_1���:

loss_1G�f@��K;5       ��]�	�>����A�
*'

learning_rate_1���:

loss_1�gJ@4V�5       ��]�	C�����A�
*'

learning_rate_1���:

loss_1�\c@�Ҏ�5       ��]�	v���A�
*'

learning_rate_1���:

loss_1�m@mbL�5       ��]�	��&���A�
*'

learning_rate_1���:

loss_1�Y@W���5       ��]�	ug=���A�
*'

learning_rate_1���:

loss_1�^@�,�<5       ��]�	V�T���A�
*'

learning_rate_1���:

loss_1)[@F,XK5       ��]�	=�j���A�
*'

learning_rate_1���:

loss_1L�]@���5       ��]�	�΁���A�
*'

learning_rate_1���:

loss_1k]\@lCs5       ��]�	�����A�
*'

learning_rate_1���:

loss_1me@�E��5       ��]�	.f����A�
*'

learning_rate_1���:

loss_1wR[@Yb�5       ��]�	Q�����A�
*'

learning_rate_1���:

loss_1Lf]@�=t5       ��]�	�=����A�
*'

learning_rate_1���:

loss_1��Y@�IM�5       ��]�	�r����A�
*'

learning_rate_1���:

loss_1}xl@5���5       ��]�	�����A�
*'

learning_rate_1���:

loss_1��e@O{�5       ��]�	F����A�
*'

learning_rate_1���:

loss_1$tg@�!�5       ��]�	g54���A�
*'

learning_rate_1���:

loss_1S�X@�m�5       ��]�	��J���A�
*'

learning_rate_1���:

loss_1U�`@p���5       ��]�	9b���A�
*'

learning_rate_1���:

loss_1��`@�Px>5       ��]�	��w���A�
*'

learning_rate_1���:

loss_1��]@��!k5       ��]�	�G����A�
*'

learning_rate_1���:

loss_1~�b@TD.B5       ��]�	������A�
*'

learning_rate_1���:

loss_1�R@48O?5       ��]�	Q����A�
*'

learning_rate_1���:

loss_1KM[@�f!35       ��]�	�E����A�
*'

learning_rate_1���:

loss_14lq@�z��5       ��]�	S�����A�
*'

learning_rate_1���:

loss_1��T@�YF�5       ��]�	E�����A�
*'

learning_rate_1���:

loss_1�9\@�ԫ�5       ��]�	�m���A�
*'

learning_rate_1���:

loss_1u�X@T�f[5       ��]�	}�*���A�
*'

learning_rate_1���:

loss_1�\@Vs�5       ��]�	��@���A�
*'

learning_rate_1���:

loss_1�Cd@����5       ��]�	�%W���A�
*'

learning_rate_1���:

loss_1��[@��k�5       ��]�	%�m���A�
*'

learning_rate_1���:

loss_1�W@}65       ��]�	�����A�
*'

learning_rate_1���:

loss_1�Y@hJ��5       ��]�	�!����A�
*'

learning_rate_1���:

loss_1}^c@�W�5       ��]�	�+����A�
*'

learning_rate_1���:

loss_1	�b@�م�5       ��]�	�~����A�
*'

learning_rate_1���:

loss_1��e@��95       ��]�	�[����A�
*'

learning_rate_1���:

loss_1P�V@jwg5       ��]�	<�����A�
*'

learning_rate_1���:

loss_1
lg@mk�5       ��]�	,���A�
*'

learning_rate_1���:

loss_1^@
<35       ��]�	�!���A�
*'

learning_rate_1���:

loss_1�fS@2�
p5       ��]�	�8���A�
*'

learning_rate_1���:

loss_10�c@4q}�5       ��]�	MjO���A�
*'

learning_rate_1���:

loss_1nLV@=rrl5       ��]�	N�e���A�
*'

learning_rate_1���:

loss_1�VQ@�\��5       ��]�	
�|���A�
*'

learning_rate_1���:

loss_1�b@�@��5       ��]�	Ǌ����A�
*'

learning_rate_1���:

loss_18�[@GUd�5       ��]�	;����A�
*'

learning_rate_1���:

loss_1�j@P0�05       ��]�	�����A�
*'

learning_rate_1���:

loss_1�S@�{�S5       ��]�	KM����A�
*'

learning_rate_1���:

loss_1�yY@���5       ��]�	%�����A�
*'

learning_rate_1���:

loss_1��Y@eS�5       ��]�	�����A�
*'

learning_rate_1���:

loss_1/�^@@���5       ��]�	#����A�
*'

learning_rate_1���:

loss_10 X@E��_5       ��]�	�0���A�
*'

learning_rate_1���:

loss_17�Y@/�1�5       ��]�	oGF���A�
*'

learning_rate_1���:

loss_1��m@����5       ��]�	:^���A�
*'

learning_rate_1���:

loss_1��K@�|2I5       ��]�	�t���A�
*'

learning_rate_1���:

loss_1��W@�ǃw5       ��]�	����A�
*'

learning_rate_1���:

loss_1�$I@��Q5       ��]�	�5����A�
*'

learning_rate_1���:

loss_1�k@�?:5       ��]�	�d����A�
*'

learning_rate_1���:

loss_1#W@�)��5       ��]�	������A�
*'

learning_rate_1���:

loss_1K$U@�jF_5       ��]�	�"����A�
*'

learning_rate_1���:

loss_1�(_@Ǘ5       ��]�	c����A�
*'

learning_rate_1���:

loss_1�	d@�`�5       ��]�	�����A�
*'

learning_rate_1���:

loss_1�_@d��5       ��]�	ʂ*���A�
*'

learning_rate_1���:

loss_1��S@25       ��]�	��@���A�
*'

learning_rate_1���:

loss_1t^@��S5       ��]�	dzX���A�
*'

learning_rate_1���:

loss_1ܣA@�
XY5       ��]�	�o���A�*'

learning_rate_1���:

loss_1��W@��45       ��]�	�E����A�*'

learning_rate_1���:

loss_1��d@5�H�5       ��]�	{�����A�*'

learning_rate_1���:

loss_16�Q@���5       ��]�	�+����A�*'

learning_rate_1���:

loss_1��a@�.ф5       ��]�	�����A�*'

learning_rate_1���:

loss_1o�P@D�X5       ��]�	������A�*'

learning_rate_1���:

loss_1��M@zY5       ��]�	m����A�*'

learning_rate_1���:

loss_1#*[@���5       ��]�	#`���A�*'

learning_rate_1���:

loss_1y�Y@����5       ��]�	�$���A�*'

learning_rate_1���:

loss_1P�Q@�F�_5       ��]�	)H;���A�*'

learning_rate_1���:

loss_1w�L@�\y5       ��]�	dQ���A�*'

learning_rate_1���:

loss_1��i@��^5       ��]�	�g���A�*'

learning_rate_1���:

loss_1�X@մr�5       ��]�	_~���A�*'

learning_rate_1���:

loss_1��h@"(5       ��]�	 �����A�*'

learning_rate_1���:

loss_1��L@d��M5       ��]�	������A�*'

learning_rate_1���:

loss_1�^@E�A5       ��]�	Y�����A�*'

learning_rate_1���:

loss_1��^@|��5       ��]�	_C����A�*'

learning_rate_1���:

loss_1�Y@x+�P5       ��]�	�����A�*'

learning_rate_1���:

loss_1�9W@r4��5       ��]�	A����A�*'

learning_rate_1���:

loss_1�h@��25       ��]�	�����A�*'

learning_rate_1���:

loss_1�`@�$��5       ��]�	��0���A�*'

learning_rate_1���:

loss_1�GR@&]�5       ��]�	aIG���A�*'

learning_rate_1���:

loss_1�J]@��ϒ5       ��]�	� ]���A�*'

learning_rate_1���:

loss_1r!e@�PyW5       ��]�	�>s���A�*'

learning_rate_1���:

loss_1ڧJ@{�s5       ��]�	����A�*'

learning_rate_1���:

loss_1��T@��W�5       ��]�	����A�*'

learning_rate_1���:

loss_1I�[@_��v5       ��]�	�v����A�*'

learning_rate_1���:

loss_1t Q@���F5       ��]�	������A�*'

learning_rate_1���:

loss_1��H@do��5       ��]�	������A�*'

learning_rate_1���:

loss_1��N@m�T`5       ��]�	'E����A�*'

learning_rate_1���:

loss_1�Z@�)�5       ��]�	R����A�*'

learning_rate_1���:

loss_1b�N@����5       ��]�	.�)���A�*'

learning_rate_1���:

loss_1kqX@�~�5       ��]�	��?���A�*'

learning_rate_1���:

loss_1��b@Nh�X5       ��]�	�9V���A�*'

learning_rate_1���:

loss_1U�d@م 75       ��]�	�Hl���A�*'

learning_rate_1���:

loss_1�b@�_��5       ��]�	�j����A�*'

learning_rate_1���:

loss_1U{Z@����5       ��]�	������A�*'

learning_rate_1���:

loss_1eLR@�5       ��]�	N����A�*'

learning_rate_1���:

loss_1��J@l0��5       ��]�	�Z����A�*'

learning_rate_1���:

loss_1�\[@!^��5       ��]�	&L����A�*'

learning_rate_1���:

loss_1��Y@�G�f5       ��]�	v�����A�*'

learning_rate_1���:

loss_1K J@W�5       ��]�	r�	���A�*'

learning_rate_1���:

loss_1I�Y@"t�5       ��]�	=����A�*'

learning_rate_1���:

loss_1�l@�R_�5       ��]�	6���A�*'

learning_rate_1���:

loss_1��o@Vek5       ��]�	�:L���A�*'

learning_rate_1���:

loss_1�W@��5       ��]�	g�b���A�*'

learning_rate_1���:

loss_1�V@wB�l5       ��]�	>Yz���A�*'

learning_rate_1���:

loss_1_�J@E/|F5       ��]�	������A�*'

learning_rate_1���:

loss_1�ga@�X<q5       ��]�	�j����A�*'

learning_rate_1���:

loss_19
X@�{�z5       ��]�	�]����A�*'

learning_rate_1���:

loss_1��W@Èן5       ��]�	������A�*'

learning_rate_1���:

loss_1�>f@69R�5       ��]�	������A�*'

learning_rate_1���:

loss_1��Z@�l�5       ��]�	� ���A�*'

learning_rate_1���:

loss_1�X@l��45       ��]�	�����A�*'

learning_rate_1���:

loss_1�@]@u�95       ��]�	�-���A�*'

learning_rate_1���:

loss_1�WL@�Hf�5       ��]�	IC���A�*'

learning_rate_1���:

loss_1��U@�V_45       ��]�	3Y���A�*'

learning_rate_1���:

loss_1&�R@5M�45       ��]�	��o���A�*'

learning_rate_1���:

loss_10�c@?�5       ��]�	�	����A�*'

learning_rate_1���:

loss_1VNV@�^5       ��]�	�ޛ���A�*'

learning_rate_1���:

loss_1igR@��f�5       ��]�	( ����A�*'

learning_rate_1���:

loss_1VCu@)��5       ��]�	S[����A�*'

learning_rate_1���:

loss_1,K@V���5       ��]�	P�����A�*'

learning_rate_1���:

loss_1ӷY@4�#x5       ��]�	L�����A�*'

learning_rate_1���:

loss_1��e@M���5       ��]�	����A�*'

learning_rate_1���:

loss_1p�n@ �I�5       ��]�	[!"���A�*'

learning_rate_1���:

loss_1[@���5       ��]�	�w8���A�*'

learning_rate_1���:

loss_12�^@�]#�5       ��]�	F�N���A�*'

learning_rate_1���:

loss_1�Y@tO5       ��]�	��d���A�*'

learning_rate_1���:

loss_1�bZ@���5       ��]�	�'{���A�*'

learning_rate_1���:

loss_1t!N@._#5       ��]�	�����A�*'

learning_rate_1���:

loss_1|�R@Vئ�5       ��]�	�����A�*'

learning_rate_1���:

loss_1�O@�E�5       ��]�	�c����A�*'

learning_rate_1���:

loss_1��Q@�wtG5       ��]�	������A�*'

learning_rate_1���:

loss_1v�]@�<��5       ��]�	�$����A�*'

learning_rate_1���:

loss_1�d@�IM5       ��]�	�U���A�*'

learning_rate_1���:

loss_1�U@؎��5       ��]�	ǣ���A�*'

learning_rate_1���:

loss_1+�]@.�[_5       ��]�	]�0���A�*'

learning_rate_1���:

loss_1�[@7 �5       ��]�	%FG���A�*'

learning_rate_1���:

loss_1W@�	d!5       ��]�	��]���A�*'

learning_rate_1���:

loss_14�l@�X`;5       ��]�	c�s���A�*'

learning_rate_1���:

loss_1��a@w��45       ��]�	F
����A�*'

learning_rate_1���:

loss_1�X@nō5       ��]�	�N����A�*'

learning_rate_1���:

loss_1;2c@��*5       ��]�	�Z����A�*'

learning_rate_1���:

loss_1#�N@W���5       ��]�	1�����A�*'

learning_rate_1���:

loss_1�X@9.�5       ��]�	�����A�*'

learning_rate_1���:

loss_1Cd@>�ݒ5       ��]�	�H����A�*'

learning_rate_1���:

loss_1�S@����5       ��]�	����A�*'

learning_rate_1���:

loss_1�MG@#��5       ��]�	��(���A�*'

learning_rate_1���:

loss_1�
b@�P��5       ��]�	|_?���A�*'

learning_rate_1���:

loss_1�K[@�2D�5       ��]�	=�U���A�*'

learning_rate_1���:

loss_1�)Z@��x�5       ��]�	/m���A�*'

learning_rate_1���:

loss_1Ыa@u��:5       ��]�	hu����A�*'

learning_rate_1���:

loss_1Q{`@6(D�5       ��]�	������A�*'

learning_rate_1���:

loss_1�S@���^5       ��]�	������A�*'

learning_rate_1���:

loss_1Z`W@���g5       ��]�	+e����A�*'

learning_rate_1���:

loss_1��V@�*5       ��]�	�!����A�*'

learning_rate_1���:

loss_1q�[@�g��5       ��]�	�����A�*'

learning_rate_1���:

loss_1�U@�&n"5       ��]�	�@���A�*'

learning_rate_1���:

loss_17v\@;*
5       ��]�	�!���A�*'

learning_rate_1���:

loss_1_�]@�w!5       ��]�	�7���A�*'

learning_rate_1���:

loss_1��k@4Ӡ5       ��]�	�M���A�*'

learning_rate_1���:

loss_1ئa@j�)5       ��]�	#�c���A�*'

learning_rate_1���:

loss_1�FV@0�oj5       ��]�	J�y���A�*'

learning_rate_1���:

loss_1
�N@7��G5       ��]�	������A�*'

learning_rate_1���:

loss_1�O@��5       ��]�	�5����A�*'

learning_rate_1���:

loss_1Ptm@�+�U5       ��]�	������A�*'

learning_rate_1���:

loss_1*J@�Ǣ%5       ��]�	Pd����A�*'

learning_rate_1���:

loss_1�T^@*���5       ��]�	9�����A�*'

learning_rate_1���:

loss_1��X@X��|5       ��]�	�*���A�*'

learning_rate_1���:

loss_1�K]@���5       ��]�	Ӭ���A�*'

learning_rate_1���:

loss_1�L_@kY��5       ��]�	�0���A�*'

learning_rate_1���:

loss_1��W@��"5       ��]�	[�F���A�*'

learning_rate_1���:

loss_1&P@�b��5       ��]�	�]���A�*'

learning_rate_1���:

loss_1�iO@~j�5       ��]�	fs���A�*'

learning_rate_1���:

loss_1��\@M2�R5       ��]�	t����A�*'

learning_rate_1���:

loss_1o_@��h85       ��]�	�]����A�*'

learning_rate_1���:

loss_1g�<@�S5       ��]�	Ǜ����A�*'

learning_rate_1���:

loss_1K@�G�5       ��]�	������A�*'

learning_rate_1���:

loss_1l�V@x!5       ��]�	�����A�*'

learning_rate_1���:

loss_15`@0-�5       ��]�	*I����A�*'

learning_rate_1���:

loss_1-�]@��5       ��]�	�q���A�*'

learning_rate_1���:

loss_1�U@��s�5       ��]�	$�&���A�*'

learning_rate_1���:

loss_1UMQ@��]r5       ��]�	�+=���A�*'

learning_rate_1���:

loss_1�Q@�u*y5       ��]�	
�S���A�*'

learning_rate_1���:

loss_1� O@�@X5       ��]�	M_k���A�*'

learning_rate_1���:

loss_1G+W@��+'5       ��]�	�����A�*'

learning_rate_1���:

loss_1��h@��K�5       ��]�	mG����A�*'

learning_rate_1���:

loss_1B�f@���?5       ��]�	�ܯ���A�*'

learning_rate_1���:

loss_1�P@�с,5       ��]�	+�����A�*'

learning_rate_1���:

loss_1B�P@g$�5       ��]�	������A�*'

learning_rate_1���:

loss_1ZSb@��i 5       ��]�	����A�*'

learning_rate_1���:

loss_1�`@��Pl5       ��]�	�Q	���A�*'

learning_rate_1���:

loss_1L;U@��t5       ��]�	��&���A�*'

learning_rate_1s@�:

loss_1�P@yÏ�5       ��]�	~�<���A�*'

learning_rate_1s@�:

loss_1L�I@�eZ45       ��]�	�S���A�*'

learning_rate_1s@�:

loss_1 �f@:(]�5       ��]�	� j���A�*'

learning_rate_1s@�:

loss_1��f@���5       ��]�	p�����A�*'

learning_rate_1s@�:

loss_1�$Y@���5       ��]�	�4����A�*'

learning_rate_1s@�:

loss_1ŴN@��P�5       ��]�	������A�*'

learning_rate_1s@�:

loss_1l]@A��C5       ��]�	������A�*'

learning_rate_1s@�:

loss_1^]@<o�5       ��]�	�����A�*'

learning_rate_1s@�:

loss_1Y�E@4�5|5       ��]�	������A�*'

learning_rate_1s@�:

loss_1U_U@Ǒ�5       ��]�	�� ���A�*'

learning_rate_1s@�:

loss_1�8Q@�}�5       ��]�	5�! ���A�*'

learning_rate_1s@�:

loss_1�\@>*�C5       ��]�	�}8 ���A�*'

learning_rate_1s@�:

loss_1�]@���5       ��]�	��N ���A�*'

learning_rate_1s@�:

loss_1v`@�À�5       ��]�	(�d ���A�*'

learning_rate_1s@�:

loss_1�{a@��5       ��]�	{ ���A�*'

learning_rate_1s@�:

loss_1��b@�~˥5       ��]�	�I� ���A�*'

learning_rate_1s@�:

loss_1Lm`@��C5       ��]�	�Ȩ ���A�*'

learning_rate_1s@�:

loss_1�KX@З��5       ��]�	�޿ ���A�*'

learning_rate_1s@�:

loss_1�\@F[5E5       ��]�	-�� ���A�*'

learning_rate_1s@�:

loss_1t9]@sRs5       ��]�	l�� ���A�*'

learning_rate_1s@�:

loss_1��I@�I�5       ��]�	��!���A�*'

learning_rate_1s@�:

loss_1�`@�l��5       ��]�	f!���A�*'

learning_rate_1s@�:

loss_10\@7K�5       ��]�	z�/!���A�*'

learning_rate_1s@�:

loss_1xQZ@�[��5       ��]�	��E!���A�*'

learning_rate_1s@�:

loss_1
`@�zD�5       ��]�	V�]!���A�*'

learning_rate_1s@�:

loss_1�sG@�4�/5       ��]�	N�s!���A�*'

learning_rate_1s@�:

loss_1�P@!�5       ��]�	4.�!���A�*'

learning_rate_1s@�:

loss_1�IW@m��&5       ��]�	 K�!���A�*'

learning_rate_1s@�:

loss_1�G@�gT5       ��]�	���!���A�*'

learning_rate_1s@�:

loss_1h�V@�ER�5       ��]�	K��!���A�*'

learning_rate_1s@�:

loss_1��Z@\��5       ��]�	���!���A�*'

learning_rate_1s@�:

loss_1��X@�F:r5       ��]�	��!���A�*'

learning_rate_1s@�:

loss_1��D@�Y�`5       ��]�	��"���A�*'

learning_rate_1s@�:

loss_1�MT@�-��5       ��]�	*�&"���A�*'

learning_rate_1s@�:

loss_17Q@'�?�5       ��]�	��<"���A�*'

learning_rate_1s@�:

loss_1��b@X��B5       ��]�	,!S"���A�*'

learning_rate_1s@�:

loss_1JQ@{sՇ5       ��]�	4li"���A�*'

learning_rate_1s@�:

loss_1~�d@�K�5       ��]�	�0�"���A�*'

learning_rate_1s@�:

loss_1)K@4��5       ��]�	h�"���A�*'

learning_rate_1s@�:

loss_1�^@�5       ��]�	@�"���A�*'

learning_rate_1s@�:

loss_1v�T@;Ũ5       ��]�	9D�"���A�*'

learning_rate_1s@�:

loss_1*rU@R��45       ��]�	��"���A�*'

learning_rate_1s@�:

loss_1
gM@�*�J5       ��]�	|p�"���A�*'

learning_rate_1s@�:

loss_1X/_@��?�5       ��]�	�Z#���A�*'

learning_rate_1s@�:

loss_1�h@��5       ��]�	�#���A�*'

learning_rate_1s@�:

loss_1Ĳ[@e��5       ��]�	:�5#���A�*'

learning_rate_1s@�:

loss_1�jU@�V��5       ��]�	�L#���A�*'

learning_rate_1s@�:

loss_1��X@�3�5       ��]�	�^b#���A�*'

learning_rate_1s@�:

loss_1t!Y@��<5       ��]�	Ƥx#���A�*'

learning_rate_1s@�:

loss_1�s\@�C��5       ��]�	8�#���A�*'

learning_rate_1s@�:

loss_1�^@SH�5       ��]�	���#���A�*'

learning_rate_1s@�:

loss_1�ud@Q��05       ��]�	��#���A�*'

learning_rate_1s@�:

loss_1�L`@x�b�5       ��]�	cL�#���A�*'

learning_rate_1s@�:

loss_1�a@s��5       ��]�	^�#���A�*'

learning_rate_1s@�:

loss_1k�F@��w5       ��]�	���#���A�*'

learning_rate_1s@�:

loss_1	�[@k~9\5       ��]�	>b$���A�*'

learning_rate_1s@�:

loss_1L,`@9�U�5       ��]�	^�+$���A�*'

learning_rate_1s@�:

loss_1��X@
��&5       ��]�	��A$���A�*'

learning_rate_1s@�:

loss_1w�M@jP�5       ��]�	�;X$���A�*'

learning_rate_1s@�:

loss_1oL@r��t5       ��]�	�hn$���A�*'

learning_rate_1s@�:

loss_1~JM@)�%5       ��]�	Z��$���A�*'

learning_rate_1s@�:

loss_1tW@��$5       ��]�	<��$���A�*'

learning_rate_1s@�:

loss_1��J@|CV�5       ��]�	�b�$���A�*'

learning_rate_1s@�:

loss_1�bB@����5       ��]�	m��$���A�*'

learning_rate_1s@�:

loss_1<Z@Q-^�5       ��]�	���$���A�*'

learning_rate_1s@�:

loss_1>�Y@�#��5       ��]�	X�$���A�*'

learning_rate_1s@�:

loss_1��N@����5       ��]�	j�%���A�*'

learning_rate_1s@�:

loss_1P@�Nj|5       ��]�	5�!%���A�*'

learning_rate_1s@�:

loss_1�>Z@���5       ��]�	68%���A�*'

learning_rate_1s@�:

loss_1��I@`S8q5       ��]�	SN%���A�*'

learning_rate_1s@�:

loss_1[@��Z5       ��]�	�d%���A�*'

learning_rate_1s@�:

loss_1��V@�H��5       ��]�	�/{%���A�*'

learning_rate_1s@�:

loss_1��T@�U5       ��]�	�X�%���A�*'

learning_rate_1s@�:

loss_1�	h@'n!�5       ��]�	4E�%���A�*'

learning_rate_1s@�:

loss_1��W@��"n5       ��]�	*��%���A�*'

learning_rate_1s@�:

loss_1�M@��;�5       ��]�	��%���A�*'

learning_rate_1s@�:

loss_1TRM@�E[5       ��]�	��%���A�*'

learning_rate_1s@�:

loss_1(�J@!��5       ��]�	}&���A�*'

learning_rate_1s@�:

loss_1Ec@�kƭ5       ��]�	��&���A�*'

learning_rate_1s@�:

loss_1�`@�5��5       ��]�	GP/&���A�*'

learning_rate_1s@�:

loss_1�W@��!�5       ��]�	Y3F&���A�*'

learning_rate_1s@�:

loss_1�O@���5       ��]�	��\&���A�*'

learning_rate_1s@�:

loss_1Z�Q@�m� 5       ��]�	��r&���A�*'

learning_rate_1s@�:

loss_1$�X@��3A5       ��]�	k/�&���A�*'

learning_rate_1s@�:

loss_1�fI@Ur5       ��]�	�&���A�*'

learning_rate_1s@�:

loss_1��S@ݱd5       ��]�	��&���A�*'

learning_rate_1s@�:

loss_1�=f@�4�5       ��]�	��&���A�*'

learning_rate_1s@�:

loss_1��I@m��35       ��]�	o�&���A�*'

learning_rate_1s@�:

loss_1$Q@Jἓ5       ��]�	E��&���A�*'

learning_rate_1s@�:

loss_1�#X@Xڮ�5       ��]�	��'���A�*'

learning_rate_1s@�:

loss_1ЩU@y9�5       ��]�	��%'���A�*'

learning_rate_1s@�:

loss_1�(M@���
5       ��]�	��<'���A�*'

learning_rate_1s@�:

loss_1!OH@�l�5       ��]�	xS'���A�*'

learning_rate_1s@�:

loss_12D[@T�1*5       ��]�	aVi'���A�*'

learning_rate_1s@�:

loss_1��S@qb��5       ��]�	��'���A�*'

learning_rate_1s@�:

loss_1UEc@�춲5       ��]�	_��'���A�*'

learning_rate_1s@�:

loss_1�[@a���5       ��]�	�y�'���A�*'

learning_rate_1s@�:

loss_1��d@����5       ��]�	�t�'���A�*'

learning_rate_1s@�:

loss_147?@���
5       ��]�	���'���A�*'

learning_rate_1s@�:

loss_1�KM@��Nn5       ��]�	�S�'���A�*'

learning_rate_1s@�:

loss_1�c[@w{e5       ��]�	�A(���A�*'

learning_rate_1s@�:

loss_1�Q@ڒw@5       ��]�	�(���A�*'

learning_rate_1s@�:

loss_1�W@Ӝ�5       ��]�	��3(���A�*'

learning_rate_1s@�:

loss_1\pZ@O:�5       ��]�	ӶK(���A�*'

learning_rate_1s@�:

loss_1��E@�$�5       ��]�	,ka(���A�*'

learning_rate_1s@�:

loss_1N@�x�\5       ��]�	�w(���A�*'

learning_rate_1s@�:

loss_1/)S@Μ�5       ��]�	�}�(���A�*'

learning_rate_1s@�:

loss_1.�S@���.5       ��]�	ģ(���A�*'

learning_rate_1s@�:

loss_1�[@o��	5       ��]�	B�(���A�*'

learning_rate_1s@�:

loss_14UL@���5       ��]�	f	�(���A�*'

learning_rate_1s@�:

loss_1=[@�U�5       ��]�	?�(���A�*'

learning_rate_1s@�:

loss_1��c@+T��5       ��]�	��(���A�*'

learning_rate_1s@�:

loss_1
ta@{�O5       ��]�	�)���A�*'

learning_rate_1s@�:

loss_1.hU@<��5       ��]�	9*)���A�*'

learning_rate_1s@�:

loss_1�`@�Ҧ�5       ��]�	�3@)���A�*'

learning_rate_1s@�:

loss_13S@���5       ��]�	�WV)���A�*'

learning_rate_1s@�:

loss_1ZR@$�h5       ��]�	X�l)���A�*'

learning_rate_1s@�:

loss_1�l@�� �5       ��]�	=�)���A�*'

learning_rate_1s@�:

loss_1�RY@��g(5       ��]�	h�)���A�*'

learning_rate_1s@�:

loss_1�xg@qc85       ��]�	S�)���A�*'

learning_rate_1s@�:

loss_1��Z@3J��5       ��]�	lf�)���A�*'

learning_rate_1s@�:

loss_1t	a@R�5       ��]�	���)���A�*'

learning_rate_1s@�:

loss_16aT@�./b5       ��]�	[��)���A�*'

learning_rate_1s@�:

loss_1��Y@��5       ��]�	�Q
*���A�*'

learning_rate_1s@�:

loss_1��]@��N�5       ��]�	�� *���A�*'

learning_rate_1s@�:

loss_1� S@�z��5       ��]�	��6*���A�*'

learning_rate_1s@�:

loss_1��V@���5       ��]�	��N*���A�*'

learning_rate_1s@�:

loss_1k�U@[
�5       ��]�	�	e*���A�*'

learning_rate_1s@�:

loss_1�S`@�)u�5       ��]�	�g{*���A�*'

learning_rate_1s@�:

loss_1�d@1�7�5       ��]�	Ӥ�*���A�*'

learning_rate_1s@�:

loss_1��L@�^tV5       ��]�	�\�*���A�*'

learning_rate_1s@�:

loss_1�ie@�Ǚ�5       ��]�	���*���A�*'

learning_rate_1s@�:

loss_1�H\@d�9�5       ��]�	���*���A�*'

learning_rate_1s@�:

loss_1kM@��ڊ5       ��]�	�y�*���A�*'

learning_rate_1s@�:

loss_1ʂg@�\��5       ��]�	,�+���A�*'

learning_rate_1s@�:

loss_1��U@��5       ��]�	}+���A�*'

learning_rate_1s@�:

loss_1��U@G�v5       ��]�	��2+���A�*'

learning_rate_1s@�:

loss_1c�H@C�,R5       ��]�	�WI+���A�*'

learning_rate_1s@�:

loss_1Z|]@�#5       ��]�	?�_+���A�*'

learning_rate_1s@�:

loss_1>5c@$��5       ��]�	 �u+���A�*'

learning_rate_1s@�:

loss_1�wR@�F��5       ��]�	_�+���A�*'

learning_rate_1s@�:

loss_1#�T@�*�5       ��]�	;�+���A�*'

learning_rate_1s@�:

loss_1�]@���{5       ��]�	�#�+���A�*'

learning_rate_1s@�:

loss_1�LR@ �i5       ��]�	}��+���A�*'

learning_rate_1s@�:

loss_1��Q@X*�O5       ��]�	԰�+���A�*'

learning_rate_1s@�:

loss_1GQ@�݌b5       ��]�	 �+���A�*'

learning_rate_1s@�:

loss_1U�P@��5       ��]�	{v,���A�*'

learning_rate_1s@�:

loss_1�8W@Q��d5       ��]�	�<*,���A�*'

learning_rate_1s@�:

loss_1ɷc@�<5       ��]�	�@,���A�*'

learning_rate_1s@�:

loss_1m�f@��� 5       ��]�	c�V,���A�*'

learning_rate_1s@�:

loss_1(�R@��:D5       ��]�	*�l,���A�*'

learning_rate_1s@�:

loss_1�,]@�+I5       ��]�	��,���A�*'

learning_rate_1s@�:

loss_1�b@^4�C5       ��]�	��,���A�*'

learning_rate_1s@�:

loss_1_�F@տ��5       ��]�	��,���A�*'

learning_rate_1s@�:

loss_1�O@�zn*5       ��]�	P��,���A�*'

learning_rate_1s@�:

loss_1~Hf@�湇5       ��]�	���,���A�*'

learning_rate_1s@�:

loss_1�Q@$��5       ��]�	y4�,���A�*'

learning_rate_1s@�:

loss_1Y@�J�H5       ��]�	��-���A�*'

learning_rate_1s@�:

loss_1�2Z@�^��5       ��]�	�^#-���A�*'

learning_rate_1s@�:

loss_1PiT@��5       ��]�	�9-���A�*'

learning_rate_1s@�:

loss_1v�M@��N�5       ��]�	E�O-���A�*'

learning_rate_1s@�:

loss_1S%X@*2�H5       ��]�	҇f-���A�*'

learning_rate_1s@�:

loss_1�Y@j/�S5       ��]�	)�|-���A�*'

learning_rate_1s@�:

loss_1�]Z@w��\5       ��]�	��-���A�*'

learning_rate_1s@�:

loss_1��T@�oX�5       ��]�	Ʃ-���A�*'

learning_rate_1s@�:

loss_1��W@�q��5       ��]�	��-���A�*'

learning_rate_1s@�:

loss_1	�S@���w5       ��]�	�Z�-���A�*'

learning_rate_1s@�:

loss_1+6S@�g�5       ��]�	V��-���A�*'

learning_rate_1s@�:

loss_1pQ@#�?5       ��]�	4�.���A�*'

learning_rate_1s@�:

loss_1� T@޸+5       ��]�	QQ.���A�*'

learning_rate_1s@�:

loss_1�_@�f65       ��]�	��/.���A�*'

learning_rate_1s@�:

loss_1&�_@�w@�5       ��]�	��E.���A�*'

learning_rate_1s@�:

loss_1�[@��i5       ��]�	��].���A�*'

learning_rate_1s@�:

loss_1�zV@.��5       ��]�	�t.���A�*'

learning_rate_1s@�:

loss_1�e@e�5<5       ��]�	$��.���A�*'

learning_rate_1s@�:

loss_1�?R@��85       ��]�	��.���A�*'

learning_rate_1s@�:

loss_1�Y@�ź5       ��]�	�W�.���A�*'

learning_rate_1s@�:

loss_1x>W@��5       ��]�	ƣ�.���A�*'

learning_rate_1s@�:

loss_1x�H@/�a�5       ��]�	���.���A�*'

learning_rate_1s@�:

loss_1��^@��D5       ��]�	�V�.���A�*'

learning_rate_1s@�:

loss_1��C@�~�[5       ��]�	�/���A�*'

learning_rate_1s@�:

loss_1]"Y@��UO5       ��]�	u(/���A�*'

learning_rate_1s@�:

loss_1dQ@s�G�5       ��]�	{}>/���A�*'

learning_rate_1s@�:

loss_1p�F@�?(5       ��]�	qV/���A�*'

learning_rate_1s@�:

loss_1 �P@� J�5       ��]�	u�m/���A�*'

learning_rate_1s@�:

loss_1	rO@��~5       ��]�	{D�/���A�*'

learning_rate_1s@�:

loss_1�N@ː`&5       ��]�	��/���A�*'

learning_rate_1s@�:

loss_1�R@`�45       ��]�	jΰ/���A�*'

learning_rate_1s@�:

loss_1�u`@(Y<�5       ��]�	���/���A�*'

learning_rate_1s@�:

loss_1�J@݂��5       ��]�	��/���A�*'

learning_rate_1s@�:

loss_1��a@��+5       ��]�	!��/���A�*'

learning_rate_1s@�:

loss_1��R@*��)5       ��]�	��	0���A�*'

learning_rate_1s@�:

loss_1��`@���5       ��]�	�0 0���A�*'

learning_rate_1s@�:

loss_1Q@E-~25       ��]�	
�70���A�*'

learning_rate_1s@�:

loss_1��H@xxqm5       ��]�	��M0���A�*'

learning_rate_1s@�:

loss_1��F@���5       ��]�	�,d0���A�*'

learning_rate_1s@�:

loss_1o�S@ b�H5       ��]�	��z0���A�*'

learning_rate_1s@�:

loss_1v�\@\�?�5       ��]�	U�0���A�*'

learning_rate_1s@�:

loss_1c�R@:&�5       ��]�	)C�0���A�*'

learning_rate_1s@�:

loss_1zGG@S/p�5       ��]�	t��0���A�*'

learning_rate_1s@�:

loss_1�Z@^��65       ��]�	E��0���A�*'

learning_rate_1s@�:

loss_1k�E@ξG�5       ��]�	���0���A�*'

learning_rate_1s@�:

loss_1�eb@q��w5       ��]�	K1���A�*'

learning_rate_1s@�:

loss_1]�W@�V5       ��]�	�p1���A�*'

learning_rate_1s@�:

loss_1nN@��W5       ��]�	z11���A�*'

learning_rate_1s@�:

loss_1r�G@9���5       ��]�	E\G1���A�*'

learning_rate_1s@�:

loss_1VB@�Yj�5       ��]�	��]1���A�*'

learning_rate_1s@�:

loss_1UT@�q`!5       ��]�	a�s1���A�*'

learning_rate_1s@�:

loss_1�&X@Z���5       ��]�	��1���A�*'

learning_rate_1s@�:

loss_1�qH@�çE5       ��]�	J#�1���A�*'

learning_rate_1s@�:

loss_1�e@�.�:5       ��]�	���1���A�*'

learning_rate_1s@�:

loss_1�[@"��E5       ��]�	_��1���A�*'

learning_rate_1s@�:

loss_1�aQ@��z5       ��]�	NP�1���A�*'

learning_rate_1s@�:

loss_1]�e@�
a�5       ��]�	L�1���A�*'

learning_rate_1s@�:

loss_1;?V@�<�"5       ��]�	�r2���A�*'

learning_rate_1s@�:

loss_1��Q@F`�:5       ��]�	�'2���A�*'

learning_rate_1s@�:

loss_1�dU@���;5       ��]�	�U=2���A�*'

learning_rate_1s@�:

loss_1ZFl@Q֍�5       ��]�	��S2���A�*'

learning_rate_1s@�:

loss_1��W@q��E5       ��]�	U�i2���A�*'

learning_rate_1s@�:

loss_1��N@��Ik5       ��]�	�3�2���A�*'

learning_rate_1s@�:

loss_1�x_@���5       ��]�	9s�2���A�*'

learning_rate_1s@�:

loss_1��Q@�Q�}5       ��]�	��2���A�*'

learning_rate_1s@�:

loss_1cnR@��5       ��]�	N�2���A�*'

learning_rate_1s@�:

loss_1UUD@Y�O�5       ��]�	�{�2���A�*'

learning_rate_1s@�:

loss_1v�i@���5       ��]�	,��2���A�*'

learning_rate_1s@�:

loss_1j~O@ꬲ�5       ��]�	�73���A�*'

learning_rate_1s@�:

loss_1kuG@>�`5       ��]�	�i3���A�*'

learning_rate_1s@�:

loss_1WO@�P�v5       ��]�	��43���A�*'

learning_rate_1s@�:

loss_1�KP@7��(5       ��]�	�L3���A�*'

learning_rate_1s@�:

loss_1��C@9�C5       ��]�	ưb3���A�*'

learning_rate_1s@�:

loss_1+\@�|G5       ��]�	�y3���A�*'

learning_rate_1s@�:

loss_1#�@@��o�5       ��]�	jX�3���A�*'

learning_rate_1s@�:

loss_1��U@&W5�5       ��]�	>9�3���A�*'

learning_rate_1s@�:

loss_1��R@��LH5       ��]�	�W�3���A�*'

learning_rate_1s@�:

loss_1�g@Jh�5       ��]�	���3���A�*'

learning_rate_1s@�:

loss_1��Z@�L5       ��]�	��3���A�*'

learning_rate_1s@�:

loss_1��[@%cQ5       ��]�	��3���A�*'

learning_rate_1s@�:

loss_1+V@���05       ��]�	�4���A�*'

learning_rate_1s@�:

loss_1�K@�k5       ��]�	�,4���A�*'

learning_rate_1s@�:

loss_17�V@cOil5       ��]�	�SB4���A�*'

learning_rate_1s@�:

loss_1�k@p�
B5       ��]�	ۋX4���A�*'

learning_rate_1s@�:

loss_1H�[@~r�$5       ��]�	�"o4���A�*'

learning_rate_1s@�:

loss_1��P@��o5       ��]�	bY�4���A�*'

learning_rate_1s@�:

loss_1�L@�l�g5       ��]�	4��4���A�*'

learning_rate_1s@�:

loss_1{^[@`��a5       ��]�	��4���A�*'

learning_rate_1s@�:

loss_1؏[@;��5       ��]�	p=�4���A�*'

learning_rate_1s@�:

loss_1_@�v;�5       ��]�	*��4���A�*'

learning_rate_1s@�:

loss_1_K@�t�F5       ��]�	B��4���A�*'

learning_rate_1s@�:

loss_11N@{B�5       ��]�	�5���A�*'

learning_rate_1s@�:

loss_1�I@�5�5       ��]�	��"5���A�*'

learning_rate_1s@�:

loss_1~�R@��w�5       ��]�	~�85���A�*'

learning_rate_1s@�:

loss_1�T@�O�5       ��]�	##O5���A�*'

learning_rate_1s@�:

loss_1-tc@P:B5       ��]�	ve5���A�*'

learning_rate_1s@�:

loss_1��A@k<r5       ��]�	'�{5���A�*'

learning_rate_1s@�:

loss_1Y�G@q!d5       ��]�	)��5���A�*'

learning_rate_1s@�:

loss_1>"P@Z�uZ5       ��]�	&Y�5���A�*'

learning_rate_1s@�:

loss_1��L@��} 5       ��]�	V��5���A�*'

learning_rate_1s@�:

loss_1�xE@��5       ��]�		��5���A�*'

learning_rate_1s@�:

loss_1Q�K@���5       ��]�	cf�5���A�*'

learning_rate_1s@�:

loss_1gT@�}��5       ��]�	��6���A�*'

learning_rate_1s@�:

loss_17�^@u�8B5       ��]�	 6���A�*'

learning_rate_1s@�:

loss_1��a@M(��5       ��]�	w&16���A�*'

learning_rate_1s@�:

loss_1YZA@���N5       ��]�	�gG6���A�*'

learning_rate_1s@�:

loss_1<�Y@�d	^5       ��]�	�]6���A�*'

learning_rate_1s@�:

loss_1J�U@x���5       ��]�	��u6���A�*'

learning_rate_1s@�:

loss_1P�H@=e\5       ��]�	�!�6���A�*'

learning_rate_1s@�:

loss_1��b@�G��5       ��]�	��6���A�*'

learning_rate_1s@�:

loss_1{�[@���5       ��]�	jP�6���A�*'

learning_rate_1s@�:

loss_1*M@��5       ��]�	�2�6���A�*'

learning_rate_1s@�:

loss_1��R@�O��5       ��]�	p�6���A�*'

learning_rate_1s@�:

loss_1�tO@�_�|5       ��]�	��6���A�*'

learning_rate_1s@�:

loss_1�X@���5       ��]�	�7���A�*'

learning_rate_1s@�:

loss_1��M@l��5       ��]�	<9'7���A�*'

learning_rate_1s@�:

loss_10�M@9�n�5       ��]�	|�=7���A�*'

learning_rate_1s@�:

loss_1��f@�d�5       ��]�	�!T7���A�*'

learning_rate_1s@�:

loss_1�S@4E8�5       ��]�	{:j7���A�*'

learning_rate_1s@�:

loss_1g]@:C҉5       ��]�	�P�7���A�*'

learning_rate_1s@�:

loss_1�b]@Q3��5       ��]�	+y�7���A�*'

learning_rate_1s@�:

loss_1��X@��;�5       ��]�	`�7���A�*'

learning_rate_1s@�:

loss_1�R@K��}5       ��]�	u��7���A�*'

learning_rate_1s@�:

loss_1��\@3�65       ��]�	YA�7���A�*'

learning_rate_1s@�:

loss_1�P@`$�5       ��]�	m��7���A�*'

learning_rate_1s@�:

loss_1��U@��Z5       ��]�	b8���A�*'

learning_rate_1s@�:

loss_19�G@_��E5       ��]�	ޞ8���A�*'

learning_rate_1s@�:

loss_1TO@�9�5       ��]�	�068���A�*'

learning_rate_1s@�:

loss_1�SK@�q85       ��]�	dL8���A�*'

learning_rate_1s@�:

loss_1?cP@-��:5       ��]�	��b8���A�*'

learning_rate_1s@�:

loss_1��D@��5       ��]�	�y8���A�*'

learning_rate_1s@�:

loss_1S�6@�;p�5       ��]�	�`�8���A�*'

learning_rate_1s@�:

loss_1�V@O�1y5       ��]�	4D�8���A�*'

learning_rate_1s@�:

loss_1>-V@CQ�5       ��]�	��8���A�*'

learning_rate_1s@�:

loss_1��T@��,-5       ��]�	L�8���A�*'

learning_rate_1s@�:

loss_1��N@��ش5       ��]�	F]�8���A�*'

learning_rate_1s@�:

loss_1{�F@S1N�5       ��]�	�w�8���A�*'

learning_rate_1s@�:

loss_1d�f@v}�5       ��]�	�9���A�*'

learning_rate_1s@�:

loss_1c�P@4��5       ��]�	Vx-9���A�*'

learning_rate_1s@�:

loss_1s\M@���p5       ��]�	�C9���A�*'

learning_rate_1s@�:

loss_1h�U@J~E�5       ��]�	"�Y9���A�*'

learning_rate_1s@�:

loss_1ߔO@�=ۃ5       ��]�	�@q9���A�*'

learning_rate_1s@�:

loss_1m�R@R�eG5       ��]�	�9���A�*'

learning_rate_1s@�:

loss_1IA@&���5       ��]�	�r�9���A�*'

learning_rate_1s@�:

loss_1�@W@�d�5       ��]�	���9���A�*'

learning_rate_1s@�:

loss_1�N@p9�~5       ��]�	5;�9���A�*'

learning_rate_1s@�:

loss_1�PX@#���5       ��]�	���9���A�*'

learning_rate_1s@�:

loss_1Z]h@;ך5       ��]�	���9���A�*'

learning_rate_1s@�:

loss_1�=Y@Q�!O5       ��]�	T:���A�*'

learning_rate_1s@�:

loss_1�a@�e�a5       ��]�	8z%:���A�*'

learning_rate_1s@�:

loss_18>]@���5       ��]�	Q�;:���A�*'

learning_rate_1s@�:

loss_1L8g@4�55       ��]�	21R:���A�*'

learning_rate_1s@�:

loss_1H�S@̧|�5       ��]�	��h:���A�*'

learning_rate_1s@�:

loss_1�\@̮�u5       ��]�	O�~:���A�*'

learning_rate_1s@�:

loss_1`�O@a�.5       ��]�	6�:���A�*'

learning_rate_1s@�:

loss_1@�H@@�65       ��]�	�F�:���A�*'

learning_rate_1s@�:

loss_1�S@��5       ��]�	���:���A�*'

learning_rate_1s@�:

loss_1�ua@���5       ��]�	���:���A�*'

learning_rate_1s@�:

loss_1��F@hT~5       ��]�	Y�:���A�*'

learning_rate_1s@�:

loss_1D�\@�l�\5       ��]�	��;���A�*'

learning_rate_1s@�:

loss_1��_@ZUq5       ��]�	T.;���A�*'

learning_rate_1s@�:

loss_1�N@��5       ��]�	�@1;���A�*'

learning_rate_1s@�:

loss_1_�U@hKt�5       ��]�	%�G;���A�*'

learning_rate_1s@�:

loss_1P�P@>.�5       ��]�	pz^;���A�*'

learning_rate_1s@�:

loss_1�CH@�z��5       ��]�	 �t;���A�*'

learning_rate_1s@�:

loss_1
�J@�-(�5       ��]�	?j�;���A�*'

learning_rate_1s@�:

loss_1�Z@kqND5       ��]�	��;���A�*'

learning_rate_1s@�:

loss_1ѲI@�G5       ��]�	���;���A�*'

learning_rate_1s@�:

loss_1�O@@5�
55       ��]�	�)�;���A�*'

learning_rate_1s@�:

loss_1**O@�D"�5       ��]�	j�;���A�*'

learning_rate_1s@�:

loss_1��S@�+Qb5       ��]�	�j�;���A�*'

learning_rate_1s@�:

loss_1�pN@>mg5       ��]�	E�<���A�*'

learning_rate_1s@�:

loss_1R�Z@�,�o5       ��]�	E(<���A�*'

learning_rate_1s@�:

loss_1�N@ZT�5       ��]�	�><���A�*'

learning_rate_1s@�:

loss_1�XT@h���5       ��]�	j�T<���A�*'

learning_rate_1s@�:

loss_1��]@��B5       ��]�	�k<���A�*'

learning_rate_1s@�:

loss_1;W@k-��5       ��]�	�Á<���A�*'

learning_rate_1s@�:

loss_1��Y@���-5       ��]�	��<���A�*'

learning_rate_1s@�:

loss_1�T@�RV�5       ��]�	Aȭ<���A�*'

learning_rate_1s@�:

loss_1&�T@]�85       ��]�	h�<���A�*'

learning_rate_1s@�:

loss_1_�M@ݕ 5       ��]�	��<���A�*'

learning_rate_1s@�:

loss_1E�\@���Z5       ��]�	l��<���A�*'

learning_rate_1s@�:

loss_1#�V@N���5       ��]�	�	=���A�*'

learning_rate_1s@�:

loss_1CP@_:?)5       ��]�	��=���A�*'

learning_rate_1s@�:

loss_1֭_@^���5       ��]�	B�3=���A�*'

learning_rate_1s@�:

loss_1�h@�ޣu5       ��]�	;
J=���A�*'

learning_rate_1s@�:

loss_1�nK@޵�5       ��]�	KV`=���A�*'

learning_rate_1s@�:

loss_1S6d@�`H�5       ��]�	Hev=���A�*'

learning_rate_1s@�:

loss_1X�T@����5       ��]�	���=���A�*'

learning_rate_1s@�:

loss_1�R@�κ5       ��]�	ݯ�=���A�*'

learning_rate_1s@�:

loss_1{�Z@�IU�5       ��]�	��=���A�*'

learning_rate_1s@�:

loss_1�bA@�{35       ��]�	���=���A�*'

learning_rate_1s@�:

loss_1��T@;�z|5       ��]�	5�=���A�*'

learning_rate_1s@�:

loss_1�J@2�[�5       ��]�	@�=���A�*'

learning_rate_1s@�:

loss_1�T@h8�5       ��]�	��>���A�*'

learning_rate_1s@�:

loss_1��`@}�J5       ��]�	q�(>���A�*'

learning_rate_1s@�:

loss_1��b@�@�5       ��]�	�f?>���A�*'

learning_rate_1s@�:

loss_1��T@�˖�5       ��]�	8�V>���A�*'

learning_rate_1s@�:

loss_15�V@@�Y�5       ��]�	�Zm>���A�*'

learning_rate_1s@�:

loss_1_<R@{*�z5       ��]�	�ރ>���A�*'

learning_rate_1s@�:

loss_1��R@�n�5       ��]�	�ݝ>���A�*'

learning_rate_1s@�:

loss_1	_S@=r<45       ��]�	h>�>���A�*'

learning_rate_1s@�:

loss_1?�Q@�{[5       ��]�	֑�>���A�*'

learning_rate_1s@�:

loss_1dgG@��L�5       ��]�	���>���A�*'

learning_rate_1s@�:

loss_1ϿO@�x�5       ��]�	���>���A�*'

learning_rate_1s@�:

loss_1��L@Z[=�5       ��]�	F?���A�*'

learning_rate_1s@�:

loss_1��`@�y"5       ��]�	^#$?���A�*'

learning_rate_1s@�:

loss_1��V@�W��5       ��]�	<:?���A�*'

learning_rate_1s@�:

loss_1؜U@���U5       ��]�	�6P?���A�*'

learning_rate_1s@�:

loss_1�n@Bq�5       ��]�	"�f?���A�*'

learning_rate_1s@�:

loss_1@�G@í�5       ��]�	5R}?���A�*'

learning_rate_1s@�:

loss_1��R@���%5       ��]�	=��?���A�*'

learning_rate_1s@�:

loss_1��N@��5       ��]�	��?���A�*'

learning_rate_1s@�:

loss_1�9@7�aI5       ��]�	��?���A�*'

learning_rate_1s@�:

loss_1�Q@��z&5       ��]�	V��?���A�*'

learning_rate_1s@�:

loss_1�	Y@V��5       ��]�	
�?���A�*'

learning_rate_1s@�:

loss_1�~_@\�o�5       ��]�	J?@���A�*'

learning_rate_1s@�:

loss_1XJS@q�>=5       ��]�	�%@���A�*'

learning_rate_1s@�:

loss_1��V@����5       ��]�	��/@���A�*'

learning_rate_1s@�:

loss_1�O@���}5       ��]�	Q�E@���A�*'

learning_rate_1s@�:

loss_1^!Q@6��5       ��]�	� \@���A�*'

learning_rate_1s@�:

loss_1$�F@�Z�5       ��]�	g>r@���A�*'

learning_rate_1s@�:

loss_13�V@�� o5       ��]�	��@���A�*'

learning_rate_1s@�:

loss_1��<@CN��5       ��]�	ʜ�@���A�*'

learning_rate_1s@�:

loss_1��^@���5       ��]�	�̷@���A�*'

learning_rate_1s@�:

loss_18`f@�ܗ5       ��]�	��@���A�*'

learning_rate_1s@�:

loss_1ͨS@G�x5       ��]�	�.�@���A�*'

learning_rate_1s@�:

loss_1
/J@�E�5       ��]�	͟�@���A�*'

learning_rate_1s@�:

loss_1��J@���D5       ��]�	~�A���A�*'

learning_rate_1s@�:

loss_1��H@s Y�5       ��]�	��)A���A�*'

learning_rate_1s@�:

loss_1�X@Uk�5       ��]�	)X@A���A�*'

learning_rate_1s@�:

loss_1#QO@@�5       ��]�	e�WA���A�*'

learning_rate_1s@�:

loss_1@�G@C���5       ��]�	��oA���A�*'

learning_rate_1s@�:

loss_1�hA@'�T�5       ��]�	&�A���A�*'

learning_rate_1s@�:

loss_1XU\@g%�M5       ��]�	m�A���A�*'

learning_rate_1s@�:

loss_1`@ %z�5       ��]�	�a�A���A�*'

learning_rate_1s@�:

loss_1��^@��<�5       ��]�	�{�A���A�*'

learning_rate_1s@�:

loss_1v~M@G��5       ��]�	f��A���A�*'

learning_rate_1s@�:

loss_1/\S@��}�5       ��]�	��A���A�*'

learning_rate_1s@�:

loss_1�G@�-J~5       ��]�	�pB���A�*'

learning_rate_1s@�:

loss_1=�S@���5       ��]�	ݮ"B���A�*'

learning_rate_1s@�:

loss_1�`Z@���G5       ��]�	��8B���A�*'

learning_rate_1s@�:

loss_1~(b@�>��5       ��]�	��OB���A�*'

learning_rate_1s@�:

loss_1�"I@�뫙5       ��]�	��fB���A�*'

learning_rate_1s@�:

loss_1G9Y@᩽U5       ��]�	�}B���A�*'

learning_rate_1s@�:

loss_17W@x�in5       ��]�	�J�B���A�*'

learning_rate_1s@�:

loss_1X�K@�`=5       ��]�	PЩB���A�*'

learning_rate_1s@�:

loss_1(�B@�0��5       ��]�	��B���A�*'

learning_rate_1s@�:

loss_1��:@�*�5       ��]�	ts�B���A�*'

learning_rate_1s@�:

loss_1�M@h�Q�5       ��]�	px�B���A�*'

learning_rate_1s@�:

loss_1��c@�w� 5       ��]�	4C���A�*'

learning_rate_1s@�:

loss_1�%U@Ia�5       ��]�	
cC���A�*'

learning_rate_1s@�:

loss_1@?S@7��5       ��]�	y�0C���A�*'

learning_rate_1s@�:

loss_10#c@��G�5       ��]�	m�FC���A�*'

learning_rate_1s@�:

loss_1h�[@��k5       ��]�	1�\C���A�*'

learning_rate_1s@�:

loss_1x�U@���5       ��]�	9�rC���A�*'

learning_rate_1s@�:

loss_1��S@��E5       ��]�	�)�C���A�*'

learning_rate_1s@�:

loss_1�[N@\�ze5       ��]�	�$�C���A�*'

learning_rate_1s@�:

loss_1�6F@�HR5       ��]�	�x�C���A�*'

learning_rate_1s@�:

loss_12R@�h�E5       ��]�	�n�C���A�*'

learning_rate_1s@�:

loss_1ßL@��4�5       ��]�	"��C���A�*'

learning_rate_1s@�:

loss_1��_@�Jә5       ��]�	���C���A�*'

learning_rate_1s@�:

loss_1�.R@6$��5       ��]�	�)D���A�*'

learning_rate_1s@�:

loss_1'�[@A�P�5       ��]�	��'D���A�*'

learning_rate_1s@�:

loss_1��V@U�q5       ��]�	9>D���A�*'

learning_rate_1s@�:

loss_1[W@B��5       ��]�	�<TD���A�*'

learning_rate_1s@�:

loss_1�EW@�@Tq5       ��]�	�gjD���A�*'

learning_rate_1s@�:

loss_1�H@Z\�5       ��]�	\��D���A�*'

learning_rate_1s@�:

loss_1@_L@�	<�5       ��]�	fƖD���A�*'

learning_rate_1s@�:

loss_1�6O@xO�5       ��]�	77�D���A�*'

learning_rate_1s@�:

loss_1xI@�f��5       ��]�	���D���A�*'

learning_rate_1s@�:

loss_1��G@�T�5       ��]�	8��D���A�*'

learning_rate_1s@�:

loss_1�b@�8�`5       ��]�	M?�D���A�*'

learning_rate_1s@�:

loss_1	=]@|SK65       ��]�	f�E���A�*'

learning_rate_1s@�:

loss_1a2J@�*�u5       ��]�	��E���A�*'

learning_rate_1s@�:

loss_1��J@��#_5       ��]�	��4E���A�*'

learning_rate_1s@�:

loss_1s�O@AQ3�5       ��]�	!PKE���A�*'

learning_rate_1s@�:

loss_1��O@�%,5       ��]�	J�aE���A�*'

learning_rate_1s@�:

loss_1�V@���%5       ��]�	��wE���A�*'

learning_rate_1s@�:

loss_1�0[@J:S�5       ��]�	��E���A�*'

learning_rate_1s@�:

loss_1!yS@o�5       ��]�	�A�E���A�*'

learning_rate_1s@�:

loss_1�P@��"5       ��]�	���E���A�*'

learning_rate_1s@�:

loss_1��W@�#�#5       ��]�	���E���A�*'

learning_rate_1s@�:

loss_1��K@���5       ��]�	�f�E���A�*'

learning_rate_1s@�:

loss_1�N@�ˬ5       ��]�	<��E���A�*'

learning_rate_1s@�:

loss_1mN@[\2o5       ��]�	��F���A�*'

learning_rate_1s@�:

loss_1!�O@bc[5       ��]�	�*F���A�*'

learning_rate_1s@�:

loss_1<�K@���a5       ��]�	QAF���A�*'

learning_rate_1s@�:

loss_1~O@�+�5       ��]�	6vWF���A�*'

learning_rate_1s@�:

loss_1�O@�9��5       ��]�	 nF���A�*'

learning_rate_1s@�:

loss_1�[@��p�5       ��]�	 &�F���A�*'

learning_rate_1s@�:

loss_1�=_@���I5       ��]�	Y��F���A�*'

learning_rate_1s@�:

loss_15�@@��cN5       ��]�	#��F���A�*'

learning_rate_1s@�:

loss_1/T@�x�5       ��]�	��F���A�*'

learning_rate_1s@�:

loss_19�^@��6�5       ��]�	ix�F���A�*'

learning_rate_1s@�:

loss_1�XQ@J�:	5       ��]�	0��F���A�*'

learning_rate_1s@�:

loss_1wD@ѝ�"5       ��]�	ttG���A�*'

learning_rate_1s@�:

loss_1��S@ɏ�5       ��]�	��!G���A�*'

learning_rate_1s@�:

loss_1 �a@G\��5       ��]�	=8G���A�*'

learning_rate_1s@�:

loss_1j�W@_./�5       ��]�	��NG���A�*'

learning_rate_1s@�:

loss_1CfR@qq�5       ��]�	J�dG���A�*'

learning_rate_1s@�:

loss_1�HE@0���5       ��]�	b�|G���A�*'

learning_rate_1s@�:

loss_1rC@�q�{5       ��]�	��G���A�*'

learning_rate_1s@�:

loss_1w9X@^�1X5       ��]�	���G���A�*'

learning_rate_1s@�:

loss_1��S@�&�5       ��]�	�.�G���A�*'

learning_rate_1s@�:

loss_1`�T@��Q�5       ��]�	~s�G���A�*'

learning_rate_1s@�:

loss_19S@���5       ��]�	��G���A�*'

learning_rate_1s@�:

loss_1@c@|J�45       ��]�	��H���A�*'

learning_rate_1s@�:

loss_1֋L@���C5       ��]�	H���A�*'

learning_rate_1s@�:

loss_1��I@��5       ��]�	�/H���A�*'

learning_rate_1s@�:

loss_1rb@'���5       ��]�	�EH���A�*'

learning_rate_1s@�:

loss_1�^@�@45       ��]�	��[H���A�*'

learning_rate_1s@�:

loss_1F�P@�O�@5       ��]�	�6sH���A�*'

learning_rate_1s@�:

loss_1ǈ_@n��5       ��]�		��H���A�*'

learning_rate_1s@�:

loss_1��b@���5       ��]�	�ğH���A�*'

learning_rate_1s@�:

loss_1��N@1W05       ��]�	���H���A�*'

learning_rate_1s@�:

loss_1��N@xaN�5       ��]�	+�H���A�*'

learning_rate_1s@�:

loss_1�VT@���5       ��]�	�_�H���A�*'

learning_rate_1s@�:

loss_1$�M@X�g�5       ��]�	׆�H���A�*'

learning_rate_1s@�:

loss_1F�V@�+�5       ��]�	!�I���A�*'

learning_rate_1s@�:

loss_1L8d@�TR+5       ��]�	'$%I���A�*'

learning_rate_1s@�:

loss_1~	Z@���5       ��]�	�A;I���A�*'

learning_rate_1s@�:

loss_1y6Y@e�] 5       ��]�	I�QI���A�*'

learning_rate_1s@�:

loss_1�,L@�؊�5       ��]�	�hiI���A�*'

learning_rate_1s@�:

loss_1f�@@�؃C5       ��]�	��I���A�*'

learning_rate_1s@�:

loss_1w�]@�@5       ��]�	I|�I���A�*'

learning_rate_1s@�:

loss_1\PG@�bV5       ��]�		��I���A�*'

learning_rate_1s@�:

loss_1 �L@S��p5       ��]�	u��I���A�*'

learning_rate_1s@�:

loss_1�GM@��5       ��]�	�f�I���A�*'

learning_rate_1s@�:

loss_1_�d@��A�5       ��]�	���I���A�*'

learning_rate_1s@�:

loss_1E�S@��TA5       ��]�	�J���A�*'

learning_rate_1s@�:

loss_1'�S@v�D�5       ��]�	BAJ���A�*'

learning_rate_1s@�:

loss_1>U@����5       ��]�	/�2J���A�*'

learning_rate_1s@�:

loss_1�`X@��5       ��]�	QIJ���A�*'

learning_rate_1s@�:

loss_1�^S@��\=5       ��]�	�e_J���A�*'

learning_rate_1s@�:

loss_1��I@2���5       ��]�	YvJ���A�*'

learning_rate_1s@�:

loss_15�B@��}-5       ��]�	ύ�J���A�*'

learning_rate_1s@�:

loss_1�V@��O5       ��]�	8/�J���A�*'

learning_rate_1s@�:

loss_1�1\@i9�5       ��]�	h��J���A�*'

learning_rate_1s@�:

loss_1_�R@)u�a5       ��]�	U�J���A�*'

learning_rate_1s@�:

loss_1�^U@Ax�5       ��]�	UO�J���A�*'

learning_rate_1s@�:

loss_1�S@�Һ5       ��]�	ڡ�J���A�*'

learning_rate_1s@�:

loss_1��N@��R 5       ��]�	�(K���A�*'

learning_rate_1s@�:

loss_1��[@�="5       ��]�	�+K���A�*'

learning_rate_1s@�:

loss_1*S@���5       ��]�	��AK���A�*'

learning_rate_1s@�:

loss_1QM@�}�v5       ��]�	aXK���A�*'

learning_rate_1s@�:

loss_1*�S@�?x5       ��]�	�DnK���A�*'

learning_rate_1s@�:

loss_1��Z@��5       ��]�	��K���A�*'

learning_rate_1s@�:

loss_1ԗW@����5       ��]�	�E�K���A�*'

learning_rate_1s@�:

loss_1�_@���5       ��]�	�K���A�*'

learning_rate_1s@�:

loss_1,<@��,�5       ��]�	��K���A�*'

learning_rate_1s@�:

loss_1i]@��E�5       ��]�	���K���A�*'

learning_rate_1s@�:

loss_140b@=Ʌ�5       ��]�	!?�K���A�*'

learning_rate_1s@�:

loss_1��P@U �5       ��]�	RzL���A�*'

learning_rate_1s@�:

loss_1��\@a�z5       ��]�	��%L���A�*'

learning_rate_1s@�:

loss_1�QK@�1 /5       ��]�	�=L���A�*'

learning_rate_1s@�:

loss_13�P@0]P�5       ��]�	�-SL���A�*'

learning_rate_1s@�:

loss_1n�E@8��z5       ��]�	siL���A�*'

learning_rate_1s@�:

loss_1��R@/�8!5       ��]�	��L���A�*'

learning_rate_1s@�:

loss_1�Q@Q���5       ��]�	�N�L���A�*'

learning_rate_1s@�:

loss_1�VY@e�:5       ��]�	���L���A�*'

learning_rate_1s@�:

loss_1mhP@��
-5       ��]�	y��L���A�*'

learning_rate_1s@�:

loss_1RS@gX��5       ��]�	�R�L���A�*'

learning_rate_1s@�:

loss_1X�S@����5       ��]�	ղ�L���A�*'

learning_rate_1s@�:

loss_1
wX@m�C�5       ��]�	M���A�*'

learning_rate_1s@�:

loss_1��H@@�m�5       ��]�	��M���A�*'

learning_rate_1s@�:

loss_1J�P@�]@@5       ��]�	H�2M���A�*'

learning_rate_1s@�:

loss_1��Y@ϒ��5       ��]�	�fIM���A�*'

learning_rate_1s@�:

loss_1	�M@�|�5       ��]�	�]M���A�*'

learning_rate_1s@�:

loss_1Ȳ[@չhY5       ��]�	ȵsM���A�*'

learning_rate_1s@�:

loss_1��Q@���c5       ��]�	�	�M���A�*'

learning_rate_1s@�:

loss_1H�W@c�{5       ��]�	���M���A�*'

learning_rate_1s@�:

loss_1.f@MH5       ��]�	�G�M���A�*'

learning_rate_1s@�:

loss_1��[@�mCI5       ��]�	�L�M���A�*'

learning_rate_1s@�:

loss_1��T@�K5       ��]�	���M���A�*'

learning_rate_1s@�:

loss_1[R@1���5       ��]�	�+�M���A�*'

learning_rate_1s@�:

loss_1�nN@����5       ��]�	pN���A�*'

learning_rate_1s@�:

loss_1ZY@P�=5       ��]�	o�&N���A�*'

learning_rate_1s@�:

loss_1�G@�Nq�5       ��]�	�8=N���A�*'

learning_rate_1s@�:

loss_1��`@+�^�5       ��]�	Q�TN���A�*'

learning_rate_1s@�:

loss_1[6E@�m8;5       ��]�	C<kN���A�*'

learning_rate_1s@�:

loss_1�_?@�I5       ��]�	"<�N���A�*'

learning_rate_1s@�:

loss_12JR@�%Q5       ��]�	5��N���A�*'

learning_rate_1s@�:

loss_1�pU@�
�5       ��]�	��N���A�*'

learning_rate_1��:

loss_1?�T@�d)5       ��]�	
i�N���A�*'

learning_rate_1��:

loss_1��B@De��5       ��]�	a��N���A�*'

learning_rate_1��:

loss_1O9S@���@5       ��]�	/�N���A�*'

learning_rate_1��:

loss_1GG@�fH15       ��]�	��O���A�*'

learning_rate_1��:

loss_1�^@@��U5       ��]�	��$O���A�*'

learning_rate_1��:

loss_1P�V@+b%-5       ��]�	S;O���A�*'

learning_rate_1��:

loss_1��N@�#�5       ��]�	�OOO���A�*'

learning_rate_1��:

loss_1	
^@��5       ��]�	X�eO���A�*'

learning_rate_1��:

loss_1w�R@��.5       ��]�	B|O���A�*'

learning_rate_1��:

loss_1`F@����5       ��]�	��O���A�*'

learning_rate_1��:

loss_1�J@��5       ��]�	q0�O���A�*'

learning_rate_1��:

loss_1��V@���5       ��]�	W��O���A�*'

learning_rate_1��:

loss_1�N\@�9Rh5       ��]�	8��O���A�*'

learning_rate_1��:

loss_1��Z@i�d5       ��]�	R�O���A�*'

learning_rate_1��:

loss_1G�K@p�>5       ��]�	�>P���A�*'

learning_rate_1��:

loss_1�Q@\��5       ��]�	ydP���A�*'

learning_rate_1��:

loss_1��P@Ht;�5       ��]�	#�/P���A�*'

learning_rate_1��:

loss_1�m[@�Ĭ�5       ��]�	"MFP���A�*'

learning_rate_1��:

loss_1�T@_��5       ��]�	q�\P���A�*'

learning_rate_1��:

loss_1=�R@	d55       ��]�	�lsP���A�*'

learning_rate_1��:

loss_1�`@�8��5       ��]�	i�P���A�*'

learning_rate_1��:

loss_1��Y@�\G�5       ��]�	�u�P���A�*'

learning_rate_1��:

loss_1�T@�,�5       ��]�	1۶P���A�*'

learning_rate_1��:

loss_1��Q@ٚ�5       ��]�	1J�P���A�*'

learning_rate_1��:

loss_1��_@>��.5       ��]�	D��P���A�*'

learning_rate_1��:

loss_1=Y@m���5       ��]�	K�P���A�*'

learning_rate_1��:

loss_158@�ͼ�5       ��]�	SQ���A�*'

learning_rate_1��:

loss_1s�L@��]�5       ��]�	B�)Q���A�*'

learning_rate_1��:

loss_1^iC@�8a'5       ��]�	~J@Q���A�*'

learning_rate_1��:

loss_1��Y@	��5       ��]�	z�VQ���A�*'

learning_rate_1��:

loss_1�LP@���5       ��]�	5�lQ���A�*'

learning_rate_1��:

loss_1�%K@C�ع5       ��]�	b�Q���A�*'

learning_rate_1��:

loss_1Q�Z@Z�(5       ��]�	0`�Q���A�*'

learning_rate_1��:

loss_1	�T@���5       ��]�	I��Q���A�*'

learning_rate_1��:

loss_1�1X@o/"5       ��]�	s6�Q���A�*'

learning_rate_1��:

loss_1C3W@�uD5       ��]�	�x�Q���A�*'

learning_rate_1��:

loss_1W�Q@�ƨB5       ��]�	%�Q���A�*'

learning_rate_1��:

loss_1R�B@��o_5       ��]�	mR���A�*'

learning_rate_1��:

loss_1�WP@$�&5       ��]�	�V!R���A�*'

learning_rate_1��:

loss_1(�N@� �5       ��]�	F)8R���A�*'

learning_rate_1��:

loss_1f�M@�L�
5       ��]�	�eNR���A�*'

learning_rate_1��:

loss_1.�F@����5       ��]�	8eR���A�*'

learning_rate_1��:

loss_11�G@�VL5       ��]�	U|R���A�*'

learning_rate_1��:

loss_12WE@���[5       ��]�	�n�R���A�*'

learning_rate_1��:

loss_1�B@GX��5       ��]�	�g�R���A�*'

learning_rate_1��:

loss_1��W@��K�5       ��]�	��R���A�*'

learning_rate_1��:

loss_1]*R@4q�5       ��]�	��R���A�*'

learning_rate_1��:

loss_1�T@Lk>�5       ��]�	"#�R���A�*'

learning_rate_1��:

loss_1=�S@\�5       ��]�	T�S���A�*'

learning_rate_1��:

loss_10�G@���5       ��]�	�S���A�*'

learning_rate_1��:

loss_1��J@��O�5       ��]�	�/0S���A�*'

learning_rate_1��:

loss_1j�X@�?��5       ��]�	`{FS���A�*'

learning_rate_1��:

loss_1&�P@���S5       ��]�	��\S���A�*'

learning_rate_1��:

loss_1�ED@l��5       ��]�	�tS���A�*'

learning_rate_1��:

loss_1��P@�h�5       ��]�	��S���A�*'

learning_rate_1��:

loss_1��E@��F5       ��]�	62�S���A�*'

learning_rate_1��:

loss_1�oR@a2޹5       ��]�	���S���A�*'

learning_rate_1��:

loss_1�.S@D]5       ��]�	��S���A�*'

learning_rate_1��:

loss_1B�@@5c`�5       ��]�	���S���A�*'

learning_rate_1��:

loss_1}>E@�=��5       ��]�	���S���A�*'

learning_rate_1��:

loss_1,T@�� �5       ��]�	=�T���A�*'

learning_rate_1��:

loss_1��K@`Y�s5       ��]�	�(+T���A�*'

learning_rate_1��:

loss_1�[X@���5       ��]�	ĄAT���A�*'

learning_rate_1��:

loss_1�R@F6�I5       ��]�	��WT���A�*'

learning_rate_1��:

loss_1��?@!,��5       ��]�	��mT���A�*'

learning_rate_1��:

loss_1�V@��5       ��]�	���T���A�*'

learning_rate_1��:

loss_1!#C@�;��5       ��]�	-�T���A�*'

learning_rate_1��:

loss_1�UN@5��5       ��]�	��T���A�*'

learning_rate_1��:

loss_1�oQ@́"W5       ��]�	�W�T���A�*'

learning_rate_1��:

loss_1��T@�S�@5       ��]�	�d�T���A�*'

learning_rate_1��:

loss_1޻H@�i�5       ��]�	,��T���A�*'

learning_rate_1��:

loss_1��H@pY�5       ��]�	��
U���A�*'

learning_rate_1��:

loss_1�Z@A�15       ��]�	�� U���A�*'

learning_rate_1��:

loss_1<�K@� �5       ��]�	7U���A�*'

learning_rate_1��:

loss_1�9S@n=�5       ��]�	OMU���A�*'

learning_rate_1��:

loss_1�M@���S5       ��]�	kcU���A�*'

learning_rate_1��:

loss_1��Q@�5       ��]�	LEyU���A�*'

learning_rate_1��:

loss_1#9P@M�t 5       ��]�	�4�U���A�*'

learning_rate_1��:

loss_1�%K@&Vh5       ��]�	�{�U���A�*'

learning_rate_1��:

loss_1��P@�]25       ��]�	衼U���A�*'

learning_rate_1��:

loss_1�(E@��e�5       ��]�	I��U���A�*'

learning_rate_1��:

loss_1&]P@bhX^5       ��]�	-�U���A�*'

learning_rate_1��:

loss_1�+X@���5       ��]�	:Y�U���A�*'

learning_rate_1��:

loss_1�3a@��C�5       ��]�	1�V���A�*'

learning_rate_1��:

loss_1v>A@��\5       ��]�	&�,V���A�*'

learning_rate_1��:

loss_1H�O@����5       ��]�	\dCV���A�*'

learning_rate_1��:

loss_1�HM@��P5       ��]�	��YV���A�*'

learning_rate_1��:

loss_1��J@�$C5       ��]�	�\pV���A�*'

learning_rate_1��:

loss_1��F@@��5       ��]�	���V���A�*'

learning_rate_1��:

loss_1�yZ@d]m�5       ��]�	��V���A�*'

learning_rate_1��:

loss_1p�S@U��e5       ��]�	P,�V���A�*'

learning_rate_1��:

loss_1��S@���35       ��]�	_p�V���A�*'

learning_rate_1��:

loss_1��N@���5       ��]�	���V���A�*'

learning_rate_1��:

loss_1!4@H�_35       ��]�	�&�V���A�*'

learning_rate_1��:

loss_1�VN@X�R*5       ��]�	LZW���A�*'

learning_rate_1��:

loss_10>@g&:�5       ��]�	�$W���A�*'

learning_rate_1��:

loss_10�c@B��e5       ��]�	(<W���A�*'

learning_rate_1��:

loss_1�8T@����5       ��]�	!�RW���A�*'

learning_rate_1��:

loss_1�O@�
�5       ��]�	fiW���A�*'

learning_rate_1��:

loss_1�M@7��|5       ��]�	��~W���A�*'

learning_rate_1��:

loss_1�2]@g_%�5       ��]�	c*�W���A�*'

learning_rate_1��:

loss_1}[@��5       ��]�	L��W���A�*'

learning_rate_1��:

loss_1v�]@R��5       ��]�	��W���A�*'

learning_rate_1��:

loss_1
�H@L@I5       ��]�	�D�W���A�*'

learning_rate_1��:

loss_1�kM@��>5       ��]�	�r�W���A�*'

learning_rate_1��:

loss_1hQ@�ki5       ��]�	ԵX���A�*'

learning_rate_1��:

loss_1zZ@pU�5       ��]�	�X���A�*'

learning_rate_1��:

loss_1�uQ@�2�x5       ��]�	Rg3X���A�*'

learning_rate_1��:

loss_1�[@��"W5       ��]�	OIX���A�*'

learning_rate_1��:

loss_1�iW@2�&5       ��]�	�_X���A�*'

learning_rate_1��:

loss_1	�U@�&�5       ��]�	��uX���A�*'

learning_rate_1��:

loss_1C�T@`��5       ��]�	+�X���A�*'

learning_rate_1��:

loss_1��S@Q���5       ��]�	���X���A�*'

learning_rate_1��:

loss_1$�R@aݡ�5       ��]�	�#�X���A�*'

learning_rate_1��:

loss_1<DG@�|M�5       ��]�	h�X���A�*'

learning_rate_1��:

loss_1�7D@9��05       ��]�	�/�X���A�*'

learning_rate_1��:

loss_1[�Q@%Re5       ��]�	7&�X���A�*'

learning_rate_1��:

loss_1�a@��px5       ��]�	�;Y���A�*'

learning_rate_1��:

loss_1�dJ@st~5       ��]�	7+Y���A�*'

learning_rate_1��:

loss_1�#O@�8?�5       ��]�	v(AY���A�*'

learning_rate_1��:

loss_1�iP@�u5       ��]�	�<WY���A�*'

learning_rate_1��:

loss_1U"Y@���M5       ��]�	eymY���A�*'

learning_rate_1��:

loss_16[@�S�?5       ��]�	�7�Y���A�*'

learning_rate_1��:

loss_1~XP@>s��5       ��]�	0x�Y���A�*'

learning_rate_1��:

loss_1%^S@�}�m5       ��]�	ᳲY���A�*'

learning_rate_1��:

loss_1��L@h ��5       ��]�	_o�Y���A�*'

learning_rate_1��:

loss_1#T@��E�5       ��]�	��Y���A�*'

learning_rate_1��:

loss_1({H@�	1l5       ��]�	���Y���A�*'

learning_rate_1��:

loss_1�iT@7f��5       ��]�	r�Z���A�*'

learning_rate_1��:

loss_1��E@m�޲5       ��]�	��$Z���A�*'

learning_rate_1��:

loss_1��T@T�05       ��]�	�7;Z���A�*'

learning_rate_1��:

loss_1slJ@I��`5       ��]�	{RZ���A�*'

learning_rate_1��:

loss_1��E@��J5       ��]�	4�iZ���A�*'

learning_rate_1��:

loss_1-H@"'��5       ��]�	��Z���A�*'

learning_rate_1��:

loss_1ѤR@�/�+5       ��]�	O�Z���A�*'

learning_rate_1��:

loss_1��B@d��`5       ��]�	�}�Z���A�*'

learning_rate_1��:

loss_1��R@[�Y 5       ��]�	���Z���A�*'

learning_rate_1��:

loss_1v�K@`��C5       ��]�	���Z���A�*'

learning_rate_1��:

loss_1�!I@�z�n5       ��]�	��Z���A�*'

learning_rate_1��:

loss_1OV@-l+�5       ��]�	b)[���A�*'

learning_rate_1��:

loss_17?@ѭd5       ��]�	gY[���A�*'

learning_rate_1��:

loss_1�]@��5       ��]�	7�1[���A�*'

learning_rate_1��:

loss_1�$U@�x�5       ��]�	XI[���A�*'

learning_rate_1��:

loss_1K�R@lP��5       ��]�	��_[���A�*'

learning_rate_1��:

loss_1L�W@m'Z5       ��]�	�Mv[���A�*'

learning_rate_1��:

loss_1�R@���T5       ��]�	���[���A�*'

learning_rate_1��:

loss_1R�L@%��5       ��]�	P�[���A�*'

learning_rate_1��:

loss_1|�[@&��5       ��]�	kV�[���A�*'

learning_rate_1��:

loss_1�eF@���5       ��]�	���[���A�*'

learning_rate_1��:

loss_1 �<@Om�5       ��]�	3��[���A�*'

learning_rate_1��:

loss_1�J\@�>��5       ��]�	�J�[���A�*'

learning_rate_1��:

loss_14�M@�ok`5       ��]�	��\���A�*'

learning_rate_1��:

loss_1��`@^�]5       ��]�	�C*\���A�*'

learning_rate_1��:

loss_1E�P@��`5       ��]�	vu@\���A�*'

learning_rate_1��:

loss_1�AM@0�Q5       ��]�	|�V\���A�*'

learning_rate_1��:

loss_1��\@Q���5       ��]�	�m\���A�*'

learning_rate_1��:

loss_1d�A@Y���5       ��]�	羄\���A�*'

learning_rate_1��:

loss_1!E@6�=e5       ��]�	P�\���A�*'

learning_rate_1��:

loss_1k->@O���5       ��]�	�Ʊ\���A�*'

learning_rate_1��:

loss_1�L@���5       ��]�	Y��\���A�*'

learning_rate_1��:

loss_1
�U@ʢO�5       ��]�	��\���A�*'

learning_rate_1��:

loss_1!�Z@J�i5       ��]�	���\���A�*'

learning_rate_1��:

loss_1QN@ ��@5       ��]�	�b]���A�*'

learning_rate_1��:

loss_1�FL@�z5       ��]�	4"]���A�*'

learning_rate_1��:

loss_1�,L@��+5       ��]�	e|8]���A�*'

learning_rate_1��:

loss_1��O@�� �5       ��]�	iO]���A�*'

learning_rate_1��:

loss_1ڕ^@j>�5       ��]�	�e]���A�*'

learning_rate_1��:

loss_1�UI@_cN5       ��]�	U]|]���A�*'

learning_rate_1��:

loss_1&�J@{Y05       ��]�	��]���A�*'

learning_rate_1��:

loss_1��E@�e�5       ��]�	���]���A�*'

learning_rate_1��:

loss_1\�J@U���5       ��]�	���]���A�*'

learning_rate_1��:

loss_1kcX@��UM5       ��]�	 %�]���A�*'

learning_rate_1��:

loss_1iV@e��5       ��]�	��]���A�*'

learning_rate_1��:

loss_1VL@d(��5       ��]�	�Z^���A�*'

learning_rate_1��:

loss_1j'<@�[�5       ��]�	�u^���A�*'

learning_rate_1��:

loss_1rb@tX�5       ��]�	E�3^���A�*'

learning_rate_1��:

loss_1�2K@��[h5       ��]�	m]J^���A�*'

learning_rate_1��:

loss_1:�O@:�,�5       ��]�	�a^���A�*'

learning_rate_1��:

loss_1|bK@�G{5       ��]�	�w^���A�*'

learning_rate_1��:

loss_1 sX@r6�>5       ��]�	N��^���A�*'

learning_rate_1��:

loss_1Zz[@��N5       ��]�	���^���A�*'

learning_rate_1��:

loss_1�M@�9 �5       ��]�	�z�^���A�*'

learning_rate_1��:

loss_1Y�Q@�_b5       ��]�	8o�^���A�*'

learning_rate_1��:

loss_1��R@��?F5       ��]�	���^���A�*'

learning_rate_1��:

loss_1&O@c��5       ��]�	��^���A�*'

learning_rate_1��:

loss_1L�\@!I05       ��]�	q�_���A�*'

learning_rate_1��:

loss_1'�J@��yx5       ��]�	9A,_���A�*'

learning_rate_1��:

loss_1'�N@딺�5       ��]�	CB_���A�*'

learning_rate_1��:

loss_1!UI@%�-5       ��]�	0:X_���A�*'

learning_rate_1��:

loss_1rD@�mo�5       ��]�	Rn_���A�*'

learning_rate_1��:

loss_1�yV@8�55       ��]�	���_���A�*'

learning_rate_1��:

loss_1��S@׶^�5       ��]�	��_���A�*'

learning_rate_1��:

loss_1��C@cw��5       ��]�	~^�_���A�*'

learning_rate_1��:

loss_1��E@��5       ��]�	��_���A�*'

learning_rate_1��:

loss_1�J@yg�d5       ��]�	�:�_���A�*'

learning_rate_1��:

loss_1��K@m"^5       ��]�	���_���A�*'

learning_rate_1��:

loss_1�H@/�5       ��]�	`���A�*'

learning_rate_1��:

loss_14�O@1�F'5       ��]�	�c"`���A�*'

learning_rate_1��:

loss_1��H@$xq5       ��]�	J�9`���A�*'

learning_rate_1��:

loss_1�H@[�y5       ��]�	XP`���A�*'

learning_rate_1��:

loss_1K�G@x��5       ��]�	OOf`���A�*'

learning_rate_1��:

loss_1��N@lh{5       ��]�	�|`���A�*'

learning_rate_1��:

loss_1;lJ@4�g�5       ��]�	�x�`���A�*'

learning_rate_1��:

loss_1�Y@�2g5       ��]�	ө`���A�*'

learning_rate_1��:

loss_1q�V@ k��5       ��]�	�`���A�*'

learning_rate_1��:

loss_1q�T@�i�5       ��]�	Nw�`���A�*'

learning_rate_1��:

loss_16)R@M�ϻ5       ��]�	��`���A�*'

learning_rate_1��:

loss_1�=J@��KS5       ��]�	�#a���A�*'

learning_rate_1��:

loss_1�`D@���i5       ��]�	f�a���A�*'

learning_rate_1��:

loss_1�_@�j�@5       ��]�	�/a���A�*'

learning_rate_1��:

loss_1c[@q@�5       ��]�	�aFa���A�*'

learning_rate_1��:

loss_1��K@�5       ��]�	A�\a���A�*'

learning_rate_1��:

loss_1Bm]@��5       ��]�	0�ra���A�*'

learning_rate_1��:

loss_1��S@���5       ��]�	o{�a���A�*'

learning_rate_1��:

loss_1�V@�E��5       ��]�	sǟa���A�*'

learning_rate_1��:

loss_1��W@+�`5       ��]�	 ��a���A�*'

learning_rate_1��:

loss_1B@I�&_5       ��]�	*Z�a���A�*'

learning_rate_1��:

loss_1��M@�M:�5       ��]�	Ӂ�a���A�*'

learning_rate_1��:

loss_1֪T@�g5       ��]�	���a���A�*'

learning_rate_1��:

loss_1{OT@�~=�5       ��]�	�&b���A�*'

learning_rate_1��:

loss_1��O@��5       ��]�	��%b���A�*'

learning_rate_1��:

loss_1�@S@���5       ��]�	�=b���A�*'

learning_rate_1��:

loss_1�<@tC�5       ��]�	��Sb���A�*'

learning_rate_1��:

loss_1��S@��$5       ��]�	jb���A�*'

learning_rate_1��:

loss_1�eB@���55       ��]�	;n�b���A�*'

learning_rate_1��:

loss_1FtM@{��5       ��]�	�֖b���A�*'

learning_rate_1��:

loss_1��Q@W�5       ��]�	���b���A�*'

learning_rate_1��:

loss_1jbD@00�5       ��]�	���b���A�*'

learning_rate_1��:

loss_1��T@���\5       ��]�	p��b���A�*'

learning_rate_1��:

loss_1�`O@�Rc5       ��]�	�Y�b���A�*'

learning_rate_1��:

loss_1LT@�v�:5       ��]�	S(c���A�*'

learning_rate_1��:

loss_1�L@�J�'5       ��]�	�Xc���A�*'

learning_rate_1��:

loss_1BZ@d���5       ��]�	C�3c���A�*'

learning_rate_1��:

loss_1w�O@��)�5       ��]�	�{Jc���A�*'

learning_rate_1��:

loss_1��G@�
\5       ��]�	ܴ`c���A�*'

learning_rate_1��:

loss_15
N@��L5       ��]�	�vc���A�*'

learning_rate_1��:

loss_1#�W@�;	5       ��]�	�c���A�*'

learning_rate_1��:

loss_1�K@�R�:5       ��]�	�ãc���A�*'

learning_rate_1��:

loss_1}xT@���M5       ��]�	>�c���A�*'

learning_rate_1��:

loss_1��8@�IC5       ��]�	܁�c���A�*'

learning_rate_1��:

loss_1�uY@��5       ��]�	0#�c���A�*'

learning_rate_1��:

loss_1��J@g��5       ��]�	G��c���A�*'

learning_rate_1��:

loss_1@�>@�y�5       ��]�	x�d���A�*'

learning_rate_1��:

loss_1�4Y@��|^5       ��]�	�%,d���A�*'

learning_rate_1��:

loss_11R@�f 5       ��]�	��Bd���A�*'

learning_rate_1��:

loss_1�.F@���5       ��]�	,Zd���A�*'

learning_rate_1��:

loss_1��W@�T25       ��]�	rd���A�*'

learning_rate_1��:

loss_1��D@��Hn5       ��]�	�C�d���A�*'

learning_rate_1��:

loss_1e�N@�)[y5       ��]�	���d���A�*'

learning_rate_1��:

loss_1t�Y@i�(5       ��]�	�K�d���A�*'

learning_rate_1��:

loss_1�Y@W�5       ��]�	�6�d���A�*'

learning_rate_1��:

loss_1��J@{�)r5       ��]�	��d���A�*'

learning_rate_1��:

loss_1�X@��<5       ��]�	r+�d���A�*'

learning_rate_1��:

loss_1��V@15�5       ��]�	m8e���A�*'

learning_rate_1��:

loss_1d�I@$.��5       ��]�	h�%e���A�*'

learning_rate_1��:

loss_1�&W@�Xdw5       ��]�	�;e���A�*'

learning_rate_1��:

loss_1k^`@D���5       ��]�	��Qe���A�*'

learning_rate_1��:

loss_1�~I@ �ڴ5       ��]�	N�ge���A�*'

learning_rate_1��:

loss_1��J@�
	5       ��]�	�~e���A�*'

learning_rate_1��:

loss_12�M@���5       ��]�	W �e���A�*'

learning_rate_1��:

loss_1-�Z@���5       ��]�	3I�e���A�*'

learning_rate_1��:

loss_1J�O@s���5       ��]�	���e���A�*'

learning_rate_1��:

loss_1��T@#|U5       ��]�	���e���A�*'

learning_rate_1��:

loss_1��_@�rR5       ��]�	�e���A�*'

learning_rate_1��:

loss_1�cA@���5       ��]�	 �f���A�*'

learning_rate_1��:

loss_1AgT@
��5       ��]�	�f���A�*'

learning_rate_1��:

loss_1��E@���u5       ��]�	wT3f���A�*'

learning_rate_1��:

loss_1�bF@W���5       ��]�	Z�If���A�*'

learning_rate_1��:

loss_1��M@lL��5       ��]�	B�_f���A�*'

learning_rate_1��:

loss_1�N@�%�5       ��]�	�vf���A�*'

learning_rate_1��:

loss_1S�G@�T�%5       ��]�	�p�f���A�*'

learning_rate_1��:

loss_1.K@m��5       ��]�	gc�f���A�*'

learning_rate_1��:

loss_1~�H@z�� 5       ��]�	��f���A�*'

learning_rate_1��:

loss_1<�R@��=5       ��]�	�d�f���A�*'

learning_rate_1��:

loss_1�+H@�}�5       ��]�	��f���A�*'

learning_rate_1��:

loss_1-�R@�bŋ5       ��]�	�v�f���A�*'

learning_rate_1��:

loss_1�G@3<45       ��]�	��g���A�*'

learning_rate_1��:

loss_1�lK@P���5       ��]�	kw*g���A�*'

learning_rate_1��:

loss_1-�J@S�:�5       ��]�	`�@g���A�*'

learning_rate_1��:

loss_1_�J@E���5       ��]�	'Wg���A�*'

learning_rate_1��:

loss_1uHW@�m�5       ��]�	�Bmg���A�*'

learning_rate_1��:

loss_1Q\Y@�O%5       ��]�	sD�g���A�*'

learning_rate_1��:

loss_1�0G@�	�5       ��]�	��g���A�*'

learning_rate_1��:

loss_1\UT@o7{�5       ��]�	�+�g���A�*'

learning_rate_1��:

loss_1�W@-x�~5       ��]�	[��g���A�*'

learning_rate_1��:

loss_1m\@��B5       ��]�	=/�g���A�*'

learning_rate_1��:

loss_1dJI@\#(�5       ��]�	�|�g���A�*'

learning_rate_1��:

loss_1f�P@�s�5       ��]�	r�h���A�*'

learning_rate_1��:

loss_14lQ@���5       ��]�	�"h���A�*'

learning_rate_1��:

loss_1��Q@v��5       ��]�	��9h���A�*'

learning_rate_1��:

loss_1>_Z@�O�5       ��]�	k�Oh���A�*'

learning_rate_1��:

loss_19K@�[5       ��]�	d}fh���A�*'

learning_rate_1��:

loss_1��N@1@�5       ��]�	'�|h���A�*'

learning_rate_1��:

loss_1x�Q@��$�5       ��]�	 Œh���A�*'

learning_rate_1��:

loss_1}9X@�h�5       ��]�	�h���A�*'

learning_rate_1��:

loss_1�Q@Ҡ�	5       ��]�	�Q�h���A�*'

learning_rate_1��:

loss_1�iQ@��K�5       ��]�	IO�h���A�*'

learning_rate_1��:

loss_1�hQ@Fte�5       ��]�	��h���A�*'

learning_rate_1��:

loss_1�N@M&d5       ��]�	�i���A�*'

learning_rate_1��:

loss_19�M@�Ƌ!5       ��]�	o_i���A�*'

learning_rate_1��:

loss_1��V@!D�5       ��]�	��.i���A�*'

learning_rate_1��:

loss_1��`@b�J95       ��]�	|�Ei���A�*'

learning_rate_1��:

loss_1bB[@Ip�Y5       ��]�	�
\i���A�*'

learning_rate_1��:

loss_1FTG@N&�5       ��]�	�Dri���A�*'

learning_rate_1��:

loss_1GV@�w�45       ��]�	�B�i���A�*'

learning_rate_1��:

loss_16DX@W��5       ��]�	S��i���A�*'

learning_rate_1��:

loss_1'�K@���5       ��]�	w�i���A�*'

learning_rate_1��:

loss_1vF@��5       ��]�	A��i���A�*'

learning_rate_1��:

loss_1��E@�R�55       ��]�	�6�i���A�*'

learning_rate_1��:

loss_1��L@��5       ��]�	���i���A�*'

learning_rate_1��:

loss_1tY@�W�v5       ��]�	}&j���A�*'

learning_rate_1��:

loss_1�4_@�� 5       ��]�	v&j���A�*'

learning_rate_1��:

loss_1iP@7ƥT5       ��]�	�b<j���A�*'

learning_rate_1��:

loss_1��W@��zw5       ��]�	�Rj���A�*'

learning_rate_1��:

loss_1�L@,0*5       ��]�	��hj���A�*'

learning_rate_1��:

loss_1CVG@U'{5       ��]�	 ;j���A�*'

learning_rate_1��:

loss_1��^@�d�5       ��]�	���j���A�*'

learning_rate_1��:

loss_1��G@��5       ��]�	���j���A�*'

learning_rate_1��:

loss_1�V@-��-5       ��]�	0�j���A�*'

learning_rate_1��:

loss_1�O@T'�g5       ��]�	�
�j���A�*'

learning_rate_1��:

loss_1�D8@�DnN5       ��]�	���j���A�*'

learning_rate_1��:

loss_1�Q@�[�5       ��]�	8�k���A�*'

learning_rate_1��:

loss_1��Y@����5       ��]�	Zk���A�*'

learning_rate_1��:

loss_1]IJ@i�j�5       ��]�	�C2k���A�*'

learning_rate_1��:

loss_1V�R@D�!5       ��]�	Hk���A�*'

learning_rate_1��:

loss_1�oN@4� B5       ��]�	��]k���A�*'

learning_rate_1��:

loss_1^^R@l�T�5       ��]�	­tk���A�*'

learning_rate_1��:

loss_1=�Y@���5       ��]�	��k���A�*'

learning_rate_1��:

loss_1�@@��P5       ��]�	Y�k���A�*'

learning_rate_1��:

loss_1��P@<3��5       ��]�	d��k���A�*'

learning_rate_1��:

loss_1}�V@�8��5       ��]�	���k���A�*'

learning_rate_1��:

loss_1X�2@�l�V5       ��]�	���k���A�*'

learning_rate_1��:

loss_1��U@ГLo5       ��]�	�=�k���A�*'

learning_rate_1��:

loss_15qH@dwXR5       ��]�	�El���A�*'

learning_rate_1��:

loss_1JB>@d6�N5       ��]�	S�(l���A�*'

learning_rate_1��:

loss_1��[@p��|5       ��]�	:?l���A�*'

learning_rate_1��:

loss_1
�H@QVN5       ��]�	xSUl���A�*'

learning_rate_1��:

loss_1�DB@��|W5       ��]�	Կkl���A�*'

learning_rate_1��:

loss_1�HI@O�5       ��]�	�*�l���A�*'

learning_rate_1��:

loss_1#I@_���5       ��]�	m2�l���A�*'

learning_rate_1��:

loss_14{V@8<n�5       ��]�	��l���A�*'

learning_rate_1��:

loss_1H@0��5       ��]�	��l���A�*'

learning_rate_1��:

loss_1��U@�p��5       ��]�	�e�l���A�*'

learning_rate_1��:

loss_1=-G@����5       ��]�	��l���A�*'

learning_rate_1��:

loss_1HK@/�I�5       ��]�	��m���A�*'

learning_rate_1��:

loss_1P+U@�a5       ��]�	��m���A�*'

learning_rate_1��:

loss_1��T@?�i5       ��]�	� 4m���A�*'

learning_rate_1��:

loss_1��T@��{b5       ��]�		oJm���A�*'

learning_rate_1��:

loss_1%L@�{5       ��]�	ά`m���A�*'

learning_rate_1��:

loss_1�+J@�G�>5       ��]�	\wm���A�*'

learning_rate_1��:

loss_1��F@?@��5       ��]�	'+�m���A�*'

learning_rate_1��:

loss_1��Q@=��U5       ��]�	B�m���A�*'

learning_rate_1��:

loss_1
�O@E@�5       ��]�	�\�m���A�*'

learning_rate_1��:

loss_1�_@�tn5       ��]�	��m���A�*'

learning_rate_1��:

loss_1ZPO@�H5       ��]�	���m���A�*'

learning_rate_1��:

loss_1o�H@jB�m5       ��]�	^��m���A�*'

learning_rate_1��:

loss_1E`@���5       ��]�	�An���A�*'

learning_rate_1��:

loss_1��F@�ͷa5       ��]�	�,n���A�*'

learning_rate_1��:

loss_1SL@��Ģ5       ��]�	<�Bn���A�*'

learning_rate_1��:

loss_1{�C@3��G5       ��]�	�9Yn���A�*'

learning_rate_1��:

loss_1a�^@�P�5       ��]�	 ron���A�*'

learning_rate_1��:

loss_1��H@���5       ��]�	>��n���A�*'

learning_rate_1��:

loss_1��V@D���5       ��]�	+o�n���A�*'

learning_rate_1��:

loss_1��Q@$1tN5       ��]�	��n���A�*'

learning_rate_1��:

loss_1#XS@�Q�y5       ��]�	/��n���A�*'

learning_rate_1��:

loss_1�>@ᗐ5       ��]�	�(�n���A�*'

learning_rate_1��:

loss_1�}N@j.i85       ��]�	z��n���A�*'

learning_rate_1��:

loss_1EJL@7�F=5       ��]�	�o���A�*'

learning_rate_1��:

loss_13�F@�;��5       ��]�	?D#o���A�*'

learning_rate_1��:

loss_1z*L@�,N5       ��]�	�):o���A�*'

learning_rate_1��:

loss_1�TU@��5       ��]�	E�Po���A�*'

learning_rate_1��:

loss_1��P@�;@i5       ��]�	g*go���A�*'

learning_rate_1��:

loss_1v Q@�$].5       ��]�	Ƣ}o���A�*'

learning_rate_1��:

loss_1�@@�.�5       ��]�	�a�o���A�*'

learning_rate_1��:

loss_1��T@�;�5       ��]�	Cƪo���A�*'

learning_rate_1��:

loss_1�'J@R�U�5       ��]�	�n�o���A�*'

learning_rate_1��:

loss_18F@�k�5       ��]�	��o���A�*'

learning_rate_1��:

loss_1}�M@�A3Q5       ��]�	���o���A�*'

learning_rate_1��:

loss_1JuQ@�%>5       ��]�	�0p���A�*'

learning_rate_1��:

loss_1ΨB@�� 5       ��]�	A'p���A�*'

learning_rate_1��:

loss_1\�N@d�B"5       ��]�	,1p���A�*'

learning_rate_1��:

loss_1��V@B�?,5       ��]�	"xGp���A�*'

learning_rate_1��:

loss_1��T@ʑ >5       ��]�	��]p���A�*'

learning_rate_1��:

loss_10KN@�\�$5       ��]�	�;tp���A�*'

learning_rate_1��:

loss_1 Z@.a�b5       ��]�	�׋p���A�*'

learning_rate_1��:

loss_1U�G@E��5       ��]�	�#�p���A�*'

learning_rate_1��:

loss_1��I@�M�5       ��]�	{��p���A�*'

learning_rate_1��:

loss_1�D@���U5       ��]�	Ld�p���A�*'

learning_rate_1��:

loss_1W�F@=�`�5       ��]�	2X�p���A�*'

learning_rate_1��:

loss_1�R@C�i5       ��]�	h�p���A�*'

learning_rate_1��:

loss_1=O@ܗ�u5       ��]�	�lq���A�*'

learning_rate_1��:

loss_1 �R@��5       ��]�	��(q���A�*'

learning_rate_1��:

loss_1��G@j�Y�5       ��]�	D�>q���A�*'

learning_rate_1��:

loss_1|F@a��5       ��]�	`Uq���A�*'

learning_rate_1��:

loss_1��S@����5       ��]�	��kq���A�*'

learning_rate_1��:

loss_1j><@g��5       ��]�	0,�q���A�*'

learning_rate_1��:

loss_1��W@e���5       ��]�	rl�q���A�*'

learning_rate_1��:

loss_1�,D@� Hy5       ��]�	Sh�q���A�*'

learning_rate_1��:

loss_1rc_@���5       ��]�	4��q���A�*'

learning_rate_1��:

loss_1-�c@���[5       ��]�	3�q���A�*'

learning_rate_1��:

loss_10�H@���T5       ��]�	1��q���A�*'

learning_rate_1��:

loss_1�'Y@����5       ��]�	w�r���A�*'

learning_rate_1��:

loss_1kS@�'W5       ��]�	ۏr���A�*'

learning_rate_1��:

loss_1p;U@��Z�5       ��]�	��1r���A�*'

learning_rate_1��:

loss_1mB@*�B5       ��]�	��Ir���A�*'

learning_rate_1��:

loss_1��A@�͟�5       ��]�	��_r���A�*'

learning_rate_1��:

loss_1�N@h���5       ��]�	�=vr���A�*'

learning_rate_1��:

loss_1��X@оm�5       ��]�	�E�r���A�*'

learning_rate_1��:

loss_1�2N@MNER5       ��]�	��r���A�*'

learning_rate_1��:

loss_1��V@���5       ��]�	��r���A�*'

learning_rate_1��:

loss_1�}T@�+�w5       ��]�	�N�r���A�*'

learning_rate_1��:

loss_1��O@��[45       ��]�	8��r���A�*'

learning_rate_1��:

loss_1�VQ@9!�5       ��]�	'�r���A�*'

learning_rate_1��:

loss_1�J@"g�05       ��]�	c}s���A�*'

learning_rate_1��:

loss_10UM@�:Rv5       ��]�	�B)s���A�*'

learning_rate_1��:

loss_1��J@��u`5       ��]�	(�?s���A�*'

learning_rate_1��:

loss_1�[@ҡ�5       ��]�	�Vs���A�*'

learning_rate_1��:

loss_1y�Q@�RE75       ��]�	FNms���A�*'

learning_rate_1��:

loss_1��D@qq"5       ��]�	ɷ�s���A�*'

learning_rate_1��:

loss_1*A@=X�5       ��]�		�s���A�*'

learning_rate_1��:

loss_1ygN@���5       ��]�	�P�s���A�*'

learning_rate_1��:

loss_1��G@R�̯5       ��]�	���s���A�*'

learning_rate_1��:

loss_1hU@�8�)5       ��]�	f��s���A�*'

learning_rate_1��:

loss_1�XT@��#j5       ��]�	��s���A�*'

learning_rate_1��:

loss_1v�T@���E5       ��]�	KD
t���A�*'

learning_rate_1��:

loss_1�IR@r+V5       ��]�	F�!t���A�*'

learning_rate_1��:

loss_1r]U@��35       ��]�	58t���A�*'

learning_rate_1��:

loss_1�VS@S
�I5       ��]�	�Nt���A�*'

learning_rate_1��:

loss_1ʁX@\/��5       ��]�	p�dt���A�*'

learning_rate_1��:

loss_1V.C@e��Q5       ��]�	�<{t���A�*'

learning_rate_1��:

loss_1�zS@���:5       ��]�	�v�t���A�*'

learning_rate_1��:

loss_1�\C@W4��5       ��]�	��t���A�*'

learning_rate_1��:

loss_1Ba?@���5       ��]�	h`�t���A�*'

learning_rate_1��:

loss_1;|S@X�K45       ��]�	)��t���A�*'

learning_rate_1��:

loss_1�5W@I�
�5       ��]�	�>�t���A�*'

learning_rate_1��:

loss_1=�6@]�C5       ��]�	q>u���A�*'

learning_rate_1��:

loss_1�7U@��OY5       ��]�	�su���A�*'

learning_rate_1��:

loss_1_5I@�Uf�5       ��]�	�G0u���A�*'

learning_rate_1��:

loss_1X�O@l|��5       ��]�	�Fu���A�*'

learning_rate_1��:

loss_1��G@S�J|5       ��]�	G�\u���A�*'

learning_rate_1��:

loss_1�Y@��4�5       ��]�	c�ru���A�*'

learning_rate_1��:

loss_1c�I@['5       ��]�	��u���A�*'

learning_rate_1��:

loss_1� I@��X�5       ��]�	�=�u���A�*'

learning_rate_1��:

loss_1�Q@�� 5       ��]�	kҵu���A�*'

learning_rate_1��:

loss_1��K@�K�5       ��]�	:	�u���A�*'

learning_rate_1��:

loss_1y�W@��5       ��]�	3"�u���A�*'

learning_rate_1��:

loss_1urW@c!�5       ��]�	�v�u���A�*'

learning_rate_1��:

loss_1��V@,`��5       ��]�	m�v���A�*'

learning_rate_1��:

loss_1$�U@s��5       ��]�	��$v���A�*'

learning_rate_1��:

loss_1�F@�0;5       ��]�	B7;v���A�*'

learning_rate_1��:

loss_1	=C@�(�35       ��]�	ϚRv���A�*'

learning_rate_1��:

loss_1�F@ ��5       ��]�	GHiv���A�*'

learning_rate_1��:

loss_1�zW@K�o�5       ��]�	�v�v���A�*'

learning_rate_1��:

loss_1��Q@����5       ��]�	���v���A�*'

learning_rate_1��:

loss_1��X@�Xг5       ��]�	<�v���A�*'

learning_rate_1��:

loss_1��M@��5       ��]�	���v���A�*'

learning_rate_1��:

loss_1�C@�Eso5       ��]�	�1�v���A�*'

learning_rate_1��:

loss_1�cV@�%C�5       ��]�	ͩ�v���A�*'

learning_rate_1��:

loss_1|=@p�\�5       ��]�	ڻw���A�*'

learning_rate_1��:

loss_1�)K@S���5       ��]�	��w���A�*'

learning_rate_1��:

loss_1KK@0�z�5       ��]�	�\4w���A�*'

learning_rate_1��:

loss_1��B@��5       ��]�	�sJw���A�*'

learning_rate_1��:

loss_1�R@[H�5       ��]�	^�`w���A�*'

learning_rate_1��:

loss_1��N@�4K(5       ��]�	�ww���A�*'

learning_rate_1��:

loss_1��W@����5       ��]�	c��w���A�*'

learning_rate_1��:

loss_1aLA@��i5       ��]�	P�w���A�*'

learning_rate_1��:

loss_1�nK@��j�5       ��]�	8d�w���A�*'

learning_rate_1��:

loss_1�M@t�>�5       ��]�	z��w���A�*'

learning_rate_1��:

loss_1�IC@��C5       ��]�	?M�w���A�*'

learning_rate_1��:

loss_1w�E@~�p
5       ��]�	�]�w���A�*'

learning_rate_1��:

loss_1�L@l��5       ��]�	��x���A�*'

learning_rate_1��:

loss_1KlN@鍖35       ��]�	��+x���A�*'

learning_rate_1��:

loss_1��S@94�:5       ��]�	S/Bx���A�*'

learning_rate_1��:

loss_1@\@��j5       ��]�	Z9Xx���A�*'

learning_rate_1��:

loss_1p�R@Xh�5       ��]�	�[nx���A�*'

learning_rate_1��:

loss_1�;U@hw�5       ��]�	��x���A�*'

learning_rate_1��:

loss_1�3T@}05       ��]�	�k�x���A�*'

learning_rate_1��:

loss_1��N@W|��5       ��]�	���x���A�*'

learning_rate_1��:

loss_1o3J@���C5       ��]�	���x���A�*'

learning_rate_1��:

loss_1�RV@���5       ��]�	�X�x���A�*'

learning_rate_1��:

loss_1�$Q@]|�
5       ��]�	`U�x���A�*'

learning_rate_1��:

loss_1��D@�
x�5       ��]�	��y���A�*'

learning_rate_1��:

loss_1l=@n�t5       ��]�	��!y���A�*'

learning_rate_1��:

loss_1G�C@��)�5       ��]�	08y���A�*'

learning_rate_1��:

loss_1 (E@�gc5       ��]�	�oNy���A�*'

learning_rate_1��:

loss_1L�H@q�D�5       ��]�	��dy���A�*'

learning_rate_1��:

loss_1U@��:|5       ��]�	�{y���A�*'

learning_rate_1��:

loss_1y)W@�@W�5       ��]�	m�y���A�*'

learning_rate_1��:

loss_1��O@2wh�5       ��]�	Z=�y���A�*'

learning_rate_1��:

loss_1�_R@� �5       ��]�	x�y���A�*'

learning_rate_1��:

loss_1��I@܀��5       ��]�	d��y���A�*'

learning_rate_1��:

loss_1��L@�*v�5       ��]�	9�y���A�*'

learning_rate_1��:

loss_1��N@�7��5       ��]�	7Jz���A�*'

learning_rate_1��:

loss_1"XU@`S*5       ��]�	;�z���A�*'

learning_rate_1��:

loss_1t�_@\�5       ��]�	6�-z���A�*'

learning_rate_1��:

loss_1�M@[RK5       ��]�	�Dz���A�*'

learning_rate_1��:

loss_1��B@���e5       ��]�	ٹZz���A�*'

learning_rate_1��:

loss_1J�L@�R65       ��]�	��pz���A�*'

learning_rate_1��:

loss_1w�U@�A�5       ��]�	-8�z���A�*'

learning_rate_1��:

loss_1q�;@s�I5       ��]�	G��z���A�*'

learning_rate_1��:

loss_1Z^@2��5       ��]�	'$�z���A�*'

learning_rate_1��:

loss_1�L@Ӆm�5       ��]�	�~�z���A�*'

learning_rate_1��:

loss_1��L@�.6n5       ��]�	���z���A�*'

learning_rate_1��:

loss_1�D@���"5       ��]�	��z���A�*'

learning_rate_1��:

loss_1jPT@�Ӝ�5       ��]�	;O{���A�*'

learning_rate_1��:

loss_1�mA@E�0�5       ��]�	�'{���A�*'

learning_rate_1��:

loss_1�UE@(��o5       ��]�	��={���A�*'

learning_rate_1��:

loss_1��H@8�h�5       ��]�	WT{���A�*'

learning_rate_1��:

loss_16DJ@R��5       ��]�	�Xj{���A�*'

learning_rate_1��:

loss_1��G@�z�5       ��]�	�ƀ{���A�*'

learning_rate_1��:

loss_1�W@��D.5       ��]�	� �{���A�*'

learning_rate_1��:

loss_1�:U@�OP�5       ��]�	�q�{���A�*'

learning_rate_1��:

loss_1��P@9�Q5       ��]�	e��{���A�*'

learning_rate_1��:

loss_1��`@_�_�5       ��]�	���{���A�*'

learning_rate_1��:

loss_1�IG@��YZ5       ��]�	i#�{���A�*'

learning_rate_1��:

loss_19�K@9��5       ��]�	�S|���A�*'

learning_rate_1��:

loss_1�iQ@�]��5       ��]�	�l|���A�*'

learning_rate_1��:

loss_1cO@�+{�5       ��]�	��2|���A�*'

learning_rate_1��:

loss_1fAI@���t5       ��]�	� I|���A�*'

learning_rate_1��:

loss_1$�W@[�;�5       ��]�	~d_|���A�*'

learning_rate_1��:

loss_1-J]@?;��5       ��]�	��u|���A�*'

learning_rate_1��:

loss_1x�N@���@5       ��]�	x�|���A�*'

learning_rate_1��:

loss_1�M@g�v5       ��]�	�0�|���A�*'

learning_rate_1��:

loss_1�G@[�V�5       ��]�	�D�|���A�*'

learning_rate_1��:

loss_1�L@ծ�5       ��]�	���|���A�*'

learning_rate_1��:

loss_1שZ@��E25       ��]�	 �|���A�*'

learning_rate_1��:

loss_1Z�F@����5       ��]�	�'�|���A�*'

learning_rate_1��:

loss_1�uI@�X@�5       ��]�	�o}���A�*'

learning_rate_1��:

loss_1#JR@@�g5       ��]�	>r(}���A�*'

learning_rate_1��:

loss_1��B@}�5       ��]�	��>}���A�*'

learning_rate_1��:

loss_1}O@r�5       ��]�	aU}���A�*'

learning_rate_1��:

loss_1�*O@N�	15       ��]�	*�k}���A�*'

learning_rate_1��:

loss_1�<O@���5       ��]�	��}���A�*'

learning_rate_1��:

loss_1+e<@�nd^5       ��]�	�@�}���A�*'

learning_rate_1��:

loss_1F<Q@�ӭ�5       ��]�	Y��}���A�*'

learning_rate_1��:

loss_1�dW@���5       ��]�	��}���A�*'

learning_rate_1��:

loss_1��?@���5       ��]�	��}���A�*'

learning_rate_1��:

loss_1^�7@/�\�5       ��]�	VU�}���A�*'

learning_rate_1��:

loss_1DuA@��H�5       ��]�	�v
~���A�*'

learning_rate_1��:

loss_1\�V@�֨�5       ��]�	V� ~���A�*'

learning_rate_1��:

loss_1�K@|�5       ��]�	\l=~���A�*'

learning_rate_1��:

loss_1�~K@��5       ��]�	M�S~���A�*'

learning_rate_1��:

loss_1\�O@���#5       ��]�	�i~���A�*'

learning_rate_1��:

loss_1��O@�b{5       ��]�	+I�~���A�*'

learning_rate_1��:

loss_1��<@����5       ��]�	��~���A�*'

learning_rate_1��:

loss_1�G@ �	Y5       ��]�	�ˬ~���A�*'

learning_rate_1��:

loss_1��D@-���5       ��]�	���~���A�*'

learning_rate_1��:

loss_1��H@�T5       ��]�	T��~���A�*'

learning_rate_1��:

loss_1_�E@S��5       ��]�	��~���A�*'

learning_rate_1��:

loss_1�\@�r��5       ��]�	�V���A�*'

learning_rate_1��:

loss_1a�?@J��`5       ��]�	.����A�*'

learning_rate_1��:

loss_1�>P@KN6�5       ��]�	Փ8���A�*'

learning_rate_1��:

loss_1lWQ@j���5       ��]�	��N���A�*'

learning_rate_1��:

loss_1�CP@N���5       ��]�	P�d���A�*'

learning_rate_1��:

loss_1Y�P@]M_�5       ��]�	�L{���A�*'

learning_rate_1��:

loss_1��T@q�A�5       ��]�	43����A�*'

learning_rate_1��:

loss_1]�T@��z	5       ��]�	������A�*'

learning_rate_1��:

loss_1�@@��W5       ��]�	�X����A�*'

learning_rate_1��:

loss_1eMY@#�
5       ��]�	$�����A�*'

learning_rate_1��:

loss_1%�W@�UY}5       ��]�	����A�*'

learning_rate_1��:

loss_1��=@0��5       ��]�	r����A�*'

learning_rate_1��:

loss_1�)O@�{��5       ��]�	 �����A�*'

learning_rate_1��:

loss_1T-@@>XX�5       ��]�	��*����A�*'

learning_rate_1��:

loss_1��W@����5       ��]�	x�@����A�*'

learning_rate_1��:

loss_1�UJ@�"��5       ��]�	UhW����A�*'

learning_rate_1��:

loss_1��O@�C�k5       ��]�	�m����A�*'

learning_rate_1��:

loss_1�*R@�X�x5       ��]�	�<�����A�*'

learning_rate_1��:

loss_1�X@��*�5       ��]�	�v�����A�*'

learning_rate_1��:

loss_1en>@����5       ��]�	������A�*'

learning_rate_1��:

loss_1ނS@�?�5       ��]�	E�ʀ���A�*'

learning_rate_1��:

loss_1�PC@�l35       ��]�	������A�*'

learning_rate_1��:

loss_1�*Q@�}�5       ��]�	0������A�*'

learning_rate_1��:

loss_1�MT@���-5       ��]�	s7����A�*'

learning_rate_1��:

loss_1��W@`�,5       ��]�	,$����A�*'

learning_rate_1��:

loss_1��;@z���5       ��]�	�e:����A�*'

learning_rate_1��:

loss_1�U@"��5       ��]�	қP����A�*'

learning_rate_1��:

loss_1�F@B��u5       ��]�	�,g����A�*'

learning_rate_1��:

loss_1N�L@���5       ��]�	�J}����A�*'

learning_rate_1��:

loss_1kZ@�=�5       ��]�	;������A�*'

learning_rate_1��:

loss_1�JL@xu�5       ��]�	!������A�*'

learning_rate_1��:

loss_1�vH@���r5       ��]�	p������A�*'

learning_rate_1��:

loss_1��J@���[5       ��]�	'�ׁ���A�*'

learning_rate_1��:

loss_1]3Z@w�F^5       ��]�	D�����A�*'

learning_rate_1��:

loss_1(D@�P�5       ��]�	�o����A�*'

learning_rate_1��:

loss_1I�H@����5       ��]�	�����A�*'

learning_rate_1��:

loss_1ZF@�'5       ��]�	��0����A�*'

learning_rate_1��:

loss_1_ZM@i(�5       ��]�	��F����A�*'

learning_rate_1��:

loss_1�UC@£5_5       ��]�	5^����A�*'

learning_rate_1��:

loss_1�P@	.s�5       ��]�	Z8t����A�*'

learning_rate_1��:

loss_1y�H@��/5       ��]�	�~�����A�*'

learning_rate_1��:

loss_1�PJ@�d�5       ��]�	eࠂ���A�*'

learning_rate_1��:

loss_1�:B@�)a5       ��]�	�����A�*'

learning_rate_1��:

loss_1�hO@^\B�5       ��]�	�͂���A�*'

learning_rate_1��:

loss_1<�C@@��+5       ��]�	F1����A�*'

learning_rate_1��:

loss_1^J>@�t5       ��]�	)������A�*'

learning_rate_1��:

loss_1��Z@G��5       ��]�	������A�*'

learning_rate_1��:

loss_1f�T@�%.�5       ��]�	��&����A�*'

learning_rate_1��:

loss_1DZL@���"5       ��]�	ׁ<����A�*'

learning_rate_1��:

loss_1pkX@T�?�5       ��]�	
�R����A�*'

learning_rate_1��:

loss_1�N@4��y5       ��]�	$i����A�*'

learning_rate_1��:

loss_1=�N@<Z!5       ��]�	�����A�*'

learning_rate_1��:

loss_1�UH@ruz5       ��]�	3m�����A�*'

learning_rate_1��:

loss_1�kC@,�dc5       ��]�	el�����A�*'

learning_rate_1��:

loss_1��J@�v�)5       ��]�	����A�*'

learning_rate_1��:

loss_1v;U@M�y�5       ��]�	��؃���A�*'

learning_rate_1��:

loss_1t�N@���5       ��]�	e����A�*'

learning_rate_1��:

loss_1��N@�	=5       ��]�	� ����A�*'

learning_rate_1��:

loss_1=XN@d1��5       ��]�	�����A�*'

learning_rate_1��:

loss_1�A@�9"5       ��]�	C'2����A�*'

learning_rate_1��:

loss_1�qK@"��5       ��]�	s�H����A�*'

learning_rate_1��:

loss_1՟R@�٥�5       ��]�	��^����A�*'

learning_rate_1��:

loss_1��M@�=5       ��]�	�u����A�*'

learning_rate_1��:

loss_1@:R@�S��5       ��]�	`q�����A�*'

learning_rate_1��:

loss_1S�S@ǭ�5       ��]�	֡����A�*'

learning_rate_1��:

loss_1��H@��!5       ��]�	�\�����A�*'

learning_rate_1��:

loss_1jA@��5       ��]�	/�τ���A�*'

learning_rate_1��:

loss_11AS@E�=5       ��]�	�����A�*'

learning_rate_1��:

loss_1��Y@Ǭ�5       ��]�	A�����A�*'

learning_rate_1��:

loss_1QW@R��5       ��]�	������A�*'

learning_rate_1��:

loss_1B�O@�GC�5       ��]�	��(����A�*'

learning_rate_1��:

loss_1.G@�\�_5       ��]�	��>����A�*'

learning_rate_1��:

loss_1ÙC@�~�5       ��]�	�W����A�*'

learning_rate_1��:

loss_1;zG@((W5       ��]�	JWm����A�*'

learning_rate_1��:

loss_1�I@��5       ��]�	�ރ����A�*'

learning_rate_1��:

loss_1�E@7���5       ��]�	�1�����A�*'

learning_rate_1��:

loss_1W�R@��'5       ��]�	������A�*'

learning_rate_1��:

loss_1KaH@�_E5       ��]�	Zǅ���A�*'

learning_rate_1��:

loss_1�gP@ߒ��5       ��]�	c�݅���A�*'

learning_rate_1��:

loss_1�EG@x�E65       ��]�	K�����A�*'

learning_rate_1��:

loss_1/ G@���f5       ��]�	;<����A�*'

learning_rate_1��:

loss_1��@@#��5       ��]�	�"����A�*'

learning_rate_1��:

loss_1�GH@�^�5       ��]�	�9����A�*'

learning_rate_1��:

loss_1�7B@ܬm�5       ��]�	�YO����A�*'

learning_rate_1��:

loss_1�S@�Q�q5       ��]�	�f����A�*'

learning_rate_1��:

loss_1�F@��k'5       ��]�	X~����A�*'

learning_rate_1��:

loss_1�?@\e��5       ��]�	�'�����A�*'

learning_rate_1��:

loss_1�C@�-�5       ��]�	�����A�*'

learning_rate_1��:

loss_1jGB@[ N�5       ��]�	0�Æ���A�*'

learning_rate_1��:

loss_1�=@@���g5       ��]�	hsچ���A�*'

learning_rate_1��:

loss_1�J@��م5       ��]�	�����A�*'

learning_rate_1��:

loss_1�qP@5:�95       ��]�		����A�*'

learning_rate_1��:

loss_1�8Q@�]�5       ��]�	(M����A�*'

learning_rate_1��:

loss_1PnN@5       ��]�	ZM4����A�*'

learning_rate_1��:

loss_1[�F@�Iŵ5       ��]�	��K����A�*'

learning_rate_1��:

loss_1�x@@d�a5       ��]�	��a����A�*'

learning_rate_1��:

loss_1�F@���5       ��]�	{x����A�*'

learning_rate_1��:

loss_1EI@[ ��5       ��]�	������A�*'

learning_rate_1��:

loss_1p�E@��J5       ��]�	�-�����A�*'

learning_rate_1��:

loss_1O@@����5       ��]�	ӕ�����A�*'

learning_rate_1��:

loss_1z)L@s�I5       ��]�	,Ӈ���A�*'

learning_rate_1��:

loss_1rX@e�r�5       ��]�	=	����A�*'

learning_rate_1��:

loss_1ٗ\@�p]5       ��]�	�H ����A�*'

learning_rate_1��:

loss_1]-E@VB�\5       ��]�	�t����A�*'

learning_rate_1��:

loss_1*dC@�<�5       ��]�	��,����A�*'

learning_rate_1��:

loss_1�V@���5       ��]�	0YC����A�*'

learning_rate_1��:

loss_1�^L@��d�5       ��]�	­Y����A�*'

learning_rate_1��:

loss_1�lU@��!�5       ��]�	��o����A�*'

learning_rate_1��:

loss_1�Y@���5       ��]�	�������A�*'

learning_rate_1��:

loss_1��<@�T�5       ��]�	�������A�*'

learning_rate_1��:

loss_16�C@#û_5       ��]�	�������A�*'

learning_rate_1��:

loss_1�?@\-75       ��]�	�sɈ���A�*'

learning_rate_1��:

loss_1��L@I���5       ��]�	Ծ߈���A�*'

learning_rate_1��:

loss_1AUE@��^5       ��]�	0�����A�*'

learning_rate_1��:

loss_11�R@k&��5       ��]�	�W����A�*'

learning_rate_1��:

loss_1�BQ@ֶB5       ��]�	>�"����A�*'

learning_rate_1��:

loss_1�2K@z�C5       ��]�	\�8����A�*'

learning_rate_1��:

loss_1�B@p��5       ��]�	�|O����A�*'

learning_rate_1��:

loss_1�VV@ayd5       ��]�	��e����A�*'

learning_rate_1��:

loss_1��Q@�ce5       ��]�	x8|����A�*'

learning_rate_1��:

loss_1`VU@�D�q5       ��]�	e[�����A�*'

learning_rate_1��:

loss_1=�D@��bU5       ��]�	\d�����A�*'

learning_rate_1��:

loss_1 S@Q�΋5       ��]�	ϰ�����A�*'

learning_rate_1��:

loss_1W�A@���5       ��]�	�aՉ���A�*'

learning_rate_1��:

loss_1�OB@H�"5       ��]�	������A�*'

learning_rate_1��:

loss_1y�O@4�)5       ��]�	�<����A�*'

learning_rate_1��:

loss_1\�@@e��5       ��]�	G4����A�*'

learning_rate_1��:

loss_1�3K@��' 5       ��]�	7r/����A�*'

learning_rate_1��:

loss_1>J@�p��5       ��]�	��E����A�*'

learning_rate_1��:

loss_1+sB@��!+5       ��]�	��[����A�*'

learning_rate_1��:

loss_1\�=@���5       ��]�	8�q����A�*'

learning_rate_1��:

loss_12�S@I֞�5       ��]�	�e�����A�*'

learning_rate_1��:

loss_1WhT@�<W5       ��]�	#������A�*'

learning_rate_1��:

loss_1OA@a!�g5       ��]�	㲊���A�*'

learning_rate_1��:

loss_1�5d@ϳ�=5       ��]�	^�Ɋ���A�*'

learning_rate_1��:

loss_1mJJ@�j5       ��]�	��ߊ���A�*'

learning_rate_1��:

loss_1��N@[gT�5       ��]�	������A�*'

learning_rate_1��:

loss_1�\@����5       ��]�	y����A�*'

learning_rate_1��:

loss_1~�G@��W5       ��]�	�a#����A�*'

learning_rate_1��:

loss_1��R@�(�75       ��]�	��9����A�*'

learning_rate_1��:

loss_1�@L@V�1,5       ��]�	DP����A�*'

learning_rate_1��:

loss_1C�M@S+ox5       ��]�	�5f����A�*'

learning_rate_1��:

loss_1o�W@�T,5       ��]�	��|����A�*'

learning_rate_1��:

loss_1O@�+�F5       ��]�	KF�����A�*'

learning_rate_1��:

loss_1��Y@�|u5       ��]�	������A�*'

learning_rate_1��:

loss_1,�>@>���5       ��]�	h������A�*'

learning_rate_1��:

loss_1ND@����5       ��]�	D�Ջ���A�*'

learning_rate_1��:

loss_1&�Y@|�_5       ��]�	.����A�*'

learning_rate_1��:

loss_1P I@�7Ag5       ��]�	�����A�*'

learning_rate_1��:

loss_1Q�X@F_�5       ��]�	2�����A�*'

learning_rate_1��:

loss_1�C@?��5       ��]�	�d/����A�*'

learning_rate_1��:

loss_1�F@9��5       ��]�	c�E����A�*'

learning_rate_1��:

loss_1#~I@6_n5       ��]�	�[����A�*'

learning_rate_1��:

loss_1��I@�it\5       ��]�	��q����A�*'

learning_rate_1��:

loss_1��D@rs�5       ��]�	K������A�*'

learning_rate_1��:

loss_1�>@s<�5       ��]�	������A�*'

learning_rate_1��:

loss_1��;@?`��5       ��]�	,T�����A�*'

learning_rate_1��:

loss_1��G@+�'5       ��]�	�`ˌ���A�*'

learning_rate_1��:

loss_1=kV@����5       ��]�	}�����A�*'

learning_rate_1��:

loss_1��@@y�.�5       ��]�	�n�����A�*'

learning_rate_1��:

loss_1ݔ3@�C�5       ��]�	������A�*'

learning_rate_1��:

loss_1J@��5       ��]�	��$����A�*'

learning_rate_1��:

loss_1v�J@�y4{5       ��]�	 ';����A�*'

learning_rate_1��:

loss_1MCT@jP1,5       ��]�	I�Q����A�*'

learning_rate_1��:

loss_1I�O@A��&5       ��]�	�h����A�*'

learning_rate_1��:

loss_1�&G@��]�5       ��]�	V]~����A�*'

learning_rate_1��:

loss_1��J@rꗁ5       ��]�	a锍���A�*'

learning_rate_1��:

loss_1B\G@�%v�5       ��]�	������A�*'

learning_rate_1��:

loss_1UbH@�dډ5       ��]�	�m���A�*'

learning_rate_1��:

loss_1|�B@Ę�5       ��]�	��ڍ���A�*'

learning_rate_1��:

loss_1��6@���5       ��]�	������A�*'

learning_rate_1��:

loss_16FO@?^�
5       ��]�	-����A�*'

learning_rate_1��:

loss_1*_Q@9��n5       ��]�	� ����A�*'

learning_rate_1��:

loss_1�7E@1x�5       ��]�	5g6����A�*'

learning_rate_1��:

loss_1��H@w���5       ��]�	P�M����A�*'

learning_rate_1��:

loss_1^%B@-]��5       ��]�	��c����A�*'

learning_rate_1��:

loss_1r�8@�͗z5       ��]�	�]z����A�*'

learning_rate_1��:

loss_1Y@�[�5       ��]�	�������A�*'

learning_rate_1��:

loss_1��O@�a5       ��]�	�s�����A�*'

learning_rate_1��:

loss_1��C@S���5       ��]�	5������A�*'

learning_rate_1��:

loss_1��S@��1�5       ��]�	7�Ԏ���A�*'

learning_rate_1��:

loss_1mhL@z[�5       ��]�	������A�*'

learning_rate_1��:

loss_1�-L@��5       ��]�	�����A�*'

learning_rate_1��:

loss_1v�R@��(5       ��]�	������A�*'

learning_rate_1��:

loss_1��A@�p
�5       ��]�	�K/����A�*'

learning_rate_1��:

loss_1��N@a�G�5       ��]�	7F����A�*'

learning_rate_1��:

loss_1�L@���5       ��]�	�d\����A�*'

learning_rate_1��:

loss_1��M@���5       ��]�	��r����A�*'

learning_rate_1��:

loss_1��I@hu�,5       ��]�	z������A�*'

learning_rate_1��:

loss_1|<@���5       ��]�	q
�����A�*'

learning_rate_1��:

loss_1��R@�/5       ��]�	�-�����A�*'

learning_rate_1��:

loss_1�#A@�-JB5       ��]�	E�͏���A�*'

learning_rate_1��:

loss_1јM@�S�15       ��]�	:�����A�*'

learning_rate_1��:

loss_15E@� G�5       ��]�	�������A�*'

learning_rate_1��:

loss_1�-J@��G5       ��]�	΄����A�*'

learning_rate_1��:

loss_1[?@q_�j5       ��]�	2*����A�*'

learning_rate_1��:

loss_1�L6@�O[p5       ��]�	�_@����A�*'

learning_rate_1��:

loss_1�S@r��5       ��]�	�V����A�*'

learning_rate_1��:

loss_1܊>@A�J5       ��]�	�3n����A�*'

learning_rate_1��:

loss_1�'B@����5       ��]�	�g�����A�*'

learning_rate_1��:

loss_1G�:@s\�`5       ��]�	E������A�*'

learning_rate_1��:

loss_1K=V@"�u�5       ��]�	�9�����A�*'

learning_rate_1��:

loss_1�D@�d )5       ��]�	V�ǐ���A�*'

learning_rate_1��:

loss_1�Q@��4�5       ��]�	U�ݐ���A�*'

learning_rate_1��:

loss_1��Q@Т:�5       ��]�	�������A�*'

learning_rate_1��:

loss_1��W@
 5       ��]�	)7����A�*'

learning_rate_1��:

loss_1�,Q@`b�5       ��]�	k!����A�*'

learning_rate_1��:

loss_1�}\@��5 5       ��]�	�7����A�*'

learning_rate_1��:

loss_1�G@hn�05       ��]�	[�M����A�*'

learning_rate_1��:

loss_1�ma@��5       ��]�	d;d����A�*'

learning_rate_1��:

loss_1	�F@�}5       ��]�	�mz����A�*'

learning_rate_1��:

loss_1΅M@�X�5       ��]�	}Ð����A�*'

learning_rate_1��:

loss_1�J@�ceD5       ��]�	�צ����A�*'

learning_rate_1��:

loss_1F@@�o5       ��]�	Ǟ�����A�*'

learning_rate_1��:

loss_1'�F@?�g55       ��]�	�֑���A�*'

learning_rate_1��:

loss_1��L@�_�5       ��]�	z�����A�*'

learning_rate_1��:

loss_1`F@�k�5       ��]�	/�����A�*'

learning_rate_1��:

loss_1|�P@�b�X5       ��]�	������A�*'

learning_rate_1��:

loss_1D�R@HG0X5       ��]�	��0����A�*'

learning_rate_1��:

loss_1��C@3�5       ��]�	~G����A�*'

learning_rate_1��:

loss_1�zG@{���5       ��]�	�*]����A�*'

learning_rate_1��:

loss_1Z�Q@bW�L5       ��]�	n9s����A�*'

learning_rate_1��:

loss_1h%@@e@��5       ��]�	#p�����A�*'

learning_rate_1��:

loss_1��F@��m5       ��]�	b~�����A�*'

learning_rate_1��:

loss_1�3H@^)R%5       ��]�	Z������A�*'

learning_rate_1��:

loss_1�yI@��s�5       ��]�	��̒���A�*'

learning_rate_1��:

loss_1,�M@��e�5       ��]�	J#����A�*'

learning_rate_1��:

loss_1VD@-���5       ��]�	{f�����A�*'

learning_rate_1��:

loss_1!�W@~
z5       ��]�	������A�*'

learning_rate_1��:

loss_1��\@�Ю�5       ��]�	�%����A�*'

learning_rate_1��:

loss_1*�N@��o�5       ��]�	)
<����A�*'

learning_rate_1��:

loss_1��R@'�VU5       ��]�	�sS����A�*'

learning_rate_1��:

loss_1׀J@R�N�5       ��]�	��i����A�*'

learning_rate_1��:

loss_1ެO@�
�5       ��]�	�&�����A�*'

learning_rate_1��:

loss_1��V@L�^5       ��]�	qy�����A�*'

learning_rate_1��:

loss_1u�I@�<95       ��]�	H�����A�*'

learning_rate_1��:

loss_1�7V@KP��5       ��]�	�{Ó���A�*'

learning_rate_1��:

loss_1c�F@���5       ��]�	�fړ���A�*'

learning_rate_1��:

loss_1��Q@���5       ��]�	�����A�*'

learning_rate_1��:

loss_1+R@7~��5       ��]�	������A�*'

learning_rate_1��:

loss_1K\@4���5       ��]�	r�����A�*'

learning_rate_1��:

loss_1��?@���5       ��]�	7����A�*'

learning_rate_1��:

loss_1S�@@��;5       ��]�	�^M����A�*'

learning_rate_1��:

loss_1ձR@$25       ��]�	 �c����A�*'

learning_rate_1��:

loss_1�B@8Tk�5       ��]�	Z�y����A�*'

learning_rate_1��:

loss_1�S<@k
V5       ��]�	f.�����A�*'

learning_rate_1��:

loss_1��C@��n5       ��]�	^�����A�*'

learning_rate_1��:

loss_1BAF@v�c5       ��]�	6������A�*'

learning_rate_1��:

loss_1�RT@z�՝5       ��]�	?�Ҕ���A�*'

learning_rate_1��:

loss_1��`@�ڲ5       ��]�	2	����A�*'

learning_rate_1��:

loss_1_�O@�w��5       ��]�	<&�����A�*'

learning_rate_1��:

loss_1d�O@��5       ��]�	Y�����A�*'

learning_rate_1��:

loss_1�*A@9	e|5       ��]�	�%,����A�*'

learning_rate_1��:

loss_1�WO@�3u�5       ��]�	hB����A�*'

learning_rate_1��:

loss_16SC@�e�'5       ��]�	��Y����A�*'

learning_rate_1��:

loss_1��L@�s�5       ��]�	Op����A�*'

learning_rate_1��:

loss_1�-W@֏�5       ��]�	�'�����A�*'

learning_rate_1��:

loss_1�E@;�5       ��]�	l�����A�*'

learning_rate_1��:

loss_1P�N@
aT5       ��]�	ó����A�*'

learning_rate_1��:

loss_1��F@���5       ��]�	�;˕���A�*'

learning_rate_1��:

loss_1��C@ı�_5       ��]�	������A�*'

learning_rate_1��:

loss_1� N@��5       ��]�	������A�*'

learning_rate_1��:

loss_1m�D@����5       ��]�	q�����A�*'

learning_rate_1��:

loss_1�lE@Xf�5       ��]�	%U%����A�*'

learning_rate_1��:

loss_1�=@_Ҍ�5       ��]�	�b;����A�*'

learning_rate_1��:

loss_1��S@���5       ��]�	�Q����A�*'

learning_rate_1��:

loss_1-Y@
s15       ��]�	��g����A�*'

learning_rate_1��:

loss_1��V@C �5       ��]�	7~����A�*'

learning_rate_1��:

loss_1	YT@V�u5       ��]�	x�����A�*'

learning_rate_1��:

loss_13oN@���5       ��]�	�j�����A�*'

learning_rate_1��:

loss_1B�H@W�j5       ��]�	�������A�*'

learning_rate_1��:

loss_1�LS@ٟ2/5       ��]�	N`ז���A�*'

learning_rate_1��:

loss_1A�H@��*5       ��]�	�\����A�*'

learning_rate_1��:

loss_1�h:@ �B�5       ��]�	�����A�*'

learning_rate_1��:

loss_1�H@.o�5       ��]�	�-����A�*'

learning_rate_1��:

loss_1ȖO@��z5       ��]�	v�3����A�*'

learning_rate_1��:

loss_1;U@?W75       ��]�	��I����A�*'

learning_rate_1��:

loss_1�=@L��5       ��]�	��_����A�*'

learning_rate_1��:

loss_1��S@@�#}5       ��]�	t�v����A�*'

learning_rate_1��:

loss_1xQ@ǩ5       ��]�	Md�����A�*'

learning_rate_1��:

loss_1��C@#�S5       ��]�	������A�*'

learning_rate_1��:

loss_1Z�H@&CB�5       ��]�	������A�*'

learning_rate_1��:

loss_1a�G@8���5       ��]�	v&ї���A�*'

learning_rate_1��:

loss_1�O@t���5       ��]�	.O����A�*'

learning_rate_1��:

loss_14�O@���5       ��]�	�~�����A�*'

learning_rate_1��:

loss_1��L@~�
�5       ��]�	�����A�*'

learning_rate_1��:

loss_1�rK@6ٔ!5       ��]�	W*����A�*'

learning_rate_1��:

loss_1�L@����5       ��]�	�T@����A�*'

learning_rate_1��:

loss_1՝@@G�5       ��]�	y�V����A�*'

learning_rate_1��:

loss_1�aO@~���5       ��]�	��m����A�*'

learning_rate_1��:

loss_1i�5@?��)5       ��]�	߄����A�*'

learning_rate_1��:

loss_1_�C@�]]�5       ��]�	8r�����A�*'

learning_rate_1��:

loss_1�[K@��u5       ��]�	޴�����A�*'

learning_rate_1��:

loss_1j<@��D�5       ��]�	Ș���A�*'

learning_rate_1��:

loss_1�M@�?�5       ��]�	L�ޘ���A�*'

learning_rate_1��:

loss_1P�R@��5       ��]�	�������A�*'

learning_rate_1��:

loss_12�I@���25       ��]�	c����A�*'

learning_rate_1��:

loss_1 ^L@���5       ��]�	��!����A�*'

learning_rate_1��:

loss_1�JZ@	}˾5       ��]�	
�7����A�*'

learning_rate_1��:

loss_1��T@@��5       ��]�	!<N����A�*'

learning_rate_1��:

loss_1BM@���5       ��]�	�qd����A�*'

learning_rate_1��:

loss_1}G@��W�5       ��]�	�{����A�*'

learning_rate_1��:

loss_1�M@q�85       ��]�	!7�����A�*'

learning_rate_1��:

loss_1s9@AZ5       ��]�	�������A�*'

learning_rate_1��:

loss_1�GK@�~{5       ��]�	����A�*'

learning_rate_1��:

loss_17�J@���5       ��]�	n�ՙ���A�*'

learning_rate_1��:

loss_1LFO@�]��5       ��]�	�����A�*'

learning_rate_1��:

loss_1��P@��J5       ��]�	�����A�*'

learning_rate_1��:

loss_1>b9@�ݩ�5       ��]�	������A�*'

learning_rate_1��:

loss_1V�S@�!�5       ��]�	
�.����A�*'

learning_rate_1��:

loss_1��N@�ҫ�5       ��]�	��D����A�*'

learning_rate_1��:

loss_1�P<@<Y|5       ��]�	`2[����A�*'

learning_rate_1��:

loss_1L�F@��K5       ��]�	��q����A�*'

learning_rate_1��:

loss_14�J@��f5       ��]�	������A�*'

learning_rate_1��:

loss_1$Y\@��;L5       ��]�	�������A�*'

learning_rate_1��:

loss_1��V@i���5       ��]�	|ച���A�*'

learning_rate_1��:

loss_1�SE@��X5       ��]�	�˚���A�*'

learning_rate_1��:

loss_1�8^@rw%5       ��]�	������A�*'

learning_rate_1��:

loss_1�ST@�K��5       ��]�	�?�����A�*'

learning_rate_1��:

loss_1*D@���5       ��]�	5�����A�*'

learning_rate_1��:

loss_1�W]@: b�5       ��]�	� %����A�*'

learning_rate_1��:

loss_15LH@y�
5       ��]�	��;����A�*'

learning_rate_1��:

loss_1�N@��5       ��]�	��R����A�*'

learning_rate_1��:

loss_1�%V@1���5       ��]�	�i����A�*'

learning_rate_1��:

loss_1؆X@h_�5       ��]�	������A�*'

learning_rate_1��:

loss_1W�W@��v5       ��]�	$?�����A�*'

learning_rate_1��:

loss_1r6O@%�J5       ��]�	�y�����A�*'

learning_rate_1��:

loss_1�aC@�B�5       ��]�	�Û���A�*'

learning_rate_1��:

loss_1l�P@w�H5       ��]�	ڛ���A�*'

learning_rate_1��:

loss_1>@'�c;5       ��]�	fe����A�*'

learning_rate_1��:

loss_1��Y@M��5       ��]�	������A�*'

learning_rate_1��:

loss_1vYR@>g��5       ��]�	������A�*'

learning_rate_1��:

loss_1V�7@s~�5       ��]�	�4����A�*'

learning_rate_1��:

loss_1J@Y6!F5       ��]�	*�K����A�*'

learning_rate_1��:

loss_1tdG@J��5       ��]�	 �a����A�*'

learning_rate_1��:

loss_1A@7���5       ��]�	})x����A�*'

learning_rate_1��:

loss_1�G@�$�5       ��]�	�������A�*'

learning_rate_1��:

loss_15F@�kU5       ��]�	�(�����A�*'

learning_rate_1��:

loss_1��8@X�B$5       ��]�	%t�����A�*'

learning_rate_1��:

loss_1�=1@��o5       ��]�	�Ҝ���A�*'

learning_rate_1��:

loss_1
�U@���5       ��]�	U����A�*'

learning_rate_1��:

loss_1�T@��g�5       ��]�	�0�����A�*'

learning_rate_1��:

loss_1�<N@D��5       ��]�	d����A�*'

learning_rate_1��:

loss_1`^Q@0�O5       ��]�	,����A�*'

learning_rate_1��:

loss_1*lR@����5       ��]�	�C����A�*'

learning_rate_1��:

loss_1�kN@}ҾZ5       ��]�	��Z����A�*'

learning_rate_1��:

loss_1��L@�ɷ!5       ��]�	��p����A�*'

learning_rate_1��:

loss_1#�V@�V��5       ��]�	�-�����A�*'

learning_rate_1��:

loss_1��@@�W�5       ��]�	zq�����A�*'

learning_rate_1��:

loss_1��L@�1��5       ��]�	I������A�*'

learning_rate_1��:

loss_1�I@��[�5       ��]�	��ɝ���A�*'

learning_rate_1��:

loss_1J{I@NXξ5       ��]�	Gz�����A�*'

learning_rate_1��:

loss_1N�=@K$bz5       ��]�	�������A�*'

learning_rate_1��:

loss_1KP@9�(�5       ��]�	�����A�*'

learning_rate_1��:

loss_1��A@���5       ��]�	�J$����A�*'

learning_rate_1��:

loss_1�W@�:�5       ��]�	y�:����A�*'

learning_rate_1��:

loss_1~
G@� �5       ��]�	��P����A�*'

learning_rate_1��:

loss_1?�U@/i��5       ��]�	>�g����A�*'

learning_rate_1��:

loss_1��C@~�/W5       ��]�	�u����A�*'

learning_rate_1��:

loss_1F&@@e �5       ��]�	$ �����A�*'

learning_rate_1��:

loss_1�jA@\��75       ��]�	ʥ�����A�*'

learning_rate_1��:

loss_1��H@b�n�5       ��]�	LÞ���A�*'

learning_rate_1��:

loss_15�J@u�{85       ��]�	��ڞ���A�*'

learning_rate_1��:

loss_1S�3@���5       ��]�	:����A�*'

learning_rate_1��:

loss_1�K@!C�R5       ��]�	�����A�*'

learning_rate_1��:

loss_1:�F@����5       ��]�	�����A�*'

learning_rate_1��:

loss_1�K@Kmq"5       ��]�	~5����A�*'

learning_rate_1��:

loss_1q�L@�O�5       ��]�	��K����A�*'

learning_rate_1��:

loss_1�(B@��{@5       ��]�	XPc����A�*'

learning_rate_1��:

loss_1�7@Z�e�5       ��]�	ɱy����A�*'

learning_rate_1��:

loss_1��Q@�#�5       ��]�	������A�*'

learning_rate_1��:

loss_1M�I@7�� 5       ��]�	�#�����A�*'

learning_rate_1��:

loss_1E�Q@4@�5       ��]�	�)�����A�*'

learning_rate_1��:

loss_1��@@S�
�5       ��]�	҅ҟ���A�*'

learning_rate_1��:

loss_1�Q@��;@5       ��]�	�O����A�*'

learning_rate_1��:

loss_1�hG@Z�?�5       ��]�	+,�����A�*'

learning_rate_1��:

loss_1��A@%�7�5       ��]�	������A�*'

learning_rate_1��:

loss_1M�I@g�i�5       ��]�	�,����A�*'

learning_rate_1��:

loss_1S�N@��|�5       ��]�	�?B����A�*'

learning_rate_1��:

loss_1T�?@e���5       ��]�	/hX����A�*'

learning_rate_1��:

loss_1{�F@g$Y[5       ��]�	ܘn����A�*'

learning_rate_1��:

loss_1C\@>J�5       ��]�	K������A�*'

learning_rate_1��:

loss_1H6@0�5       ��]�	m�����A�*'

learning_rate_1��:

loss_1��T@�x��5       ��]�	=������A�*'

learning_rate_1��:

loss_1*_>@r>5       ��]�	��ɠ���A�*'

learning_rate_1��:

loss_1�,M@�S�5       ��]�	�����A�*'

learning_rate_1��:

loss_1�7O@��5       ��]�	(�����A�*'

learning_rate_1��:

loss_1v9]@�F:5       ��]�	�����A�*'

learning_rate_1��:

loss_1kQF@���5       ��]�	y$����A�*'

learning_rate_1��:

loss_1��A@����5       ��]�	7[:����A�*'

learning_rate_1��:

loss_1pu5@Y���5       ��]�	G�P����A�*'

learning_rate_1��:

loss_13�D@ĝ��5       ��]�	 h����A�*'

learning_rate_1��:

loss_1�':@Rn�5       ��]�	/=~����A�*'

learning_rate_1��:

loss_1��I@b7WZ5       ��]�	�\�����A�*'

learning_rate_1��:

loss_1��B@�Et�5       ��]�	՚�����A�*'

learning_rate_1��:

loss_12+X@ͱ�m5       ��]�		������A�*'

learning_rate_1��:

loss_1S@���5       ��]�	�ס���A�*'

learning_rate_1��:

loss_1+>8@�X�5       ��]�	<F����A�*'

learning_rate_1��:

loss_1N<@�X�C5       ��]�	C�����A�*'

learning_rate_1��:

loss_1z�F@{�;�5       ��]�	������A�*'

learning_rate_1��:

loss_1�O@z膾5       ��]�	�L2����A�*'

learning_rate_1��:

loss_1_�V@hi��5       ��]�	6�J����A�*'

learning_rate_1��:

loss_1�B=@ALp5       ��]�	2�a����A�*'

learning_rate_1��:

loss_1-�F@��45       ��]�	\]x����A�*'

learning_rate_1��:

loss_1+UR@H(l-5       ��]�	$̎����A�*'

learning_rate_1��:

loss_1J�Z@��^5       ��]�	u�����A�*'

learning_rate_1��:

loss_1`�@@b{}5       ��]�	�������A�*'

learning_rate_1��:

loss_1��J@o|��5       ��]�	�Ѣ���A�*'

learning_rate_1��:

loss_1X�9@{D��5       ��]�	N����A�*'

learning_rate_1��:

loss_1��P@�Y�5       ��]�	�C�����A�*'

learning_rate_1��:

loss_1)=M@��=�5       ��]�	�����A�*'

learning_rate_1��:

loss_1�eG@f���5       ��]�	��+����A�*'

learning_rate_1��:

loss_1�D>@`��5       ��]�	D6B����A�*'

learning_rate_1��:

loss_1M�W@��	�5       ��]�	�cX����A�*'

learning_rate_1��:

loss_1�gK@�\`5       ��]�	I�n����A�*'

learning_rate_1��:

loss_1�B@�_��5       ��]�	�4�����A�*'

learning_rate_1��:

loss_1��H@�%��5       ��]�	�������A�*'

learning_rate_1��:

loss_1>(K@���5       ��]�	�A�����A�*'

learning_rate_1��:

loss_1l!J@ȿTz5       ��]�	�bȣ���A�*'

learning_rate_1��:

loss_1�VI@�Q[5       ��]�	�ޣ���A�*'

learning_rate_1��:

loss_1�,K@��)Y5       ��]�	Z������A�*'

learning_rate_1��:

loss_1�%O@�/F5       ��]�	�����A�*'

learning_rate_1��:

loss_1��E@}Ƈ�5       ��]�	E�!����A�*'

learning_rate_1��:

loss_1.M@��KK5       ��]�	|08����A�*'

learning_rate_1��:

loss_1�O@�N�5       ��]�	�UN����A�*'

learning_rate_1��:

loss_1ذI@��<5       ��]�	�d����A�*'

learning_rate_1��:

loss_1�T@\���5       ��]�	UP{����A�*'

learning_rate_1��:

loss_1J�L@�8�@5       ��]�	�������A�*'

learning_rate_1��:

loss_1sHM@ �f�5       ��]�	�������A�*'

learning_rate_1��:

loss_1/P@�C�]5       ��]�	�轤���A�*'

learning_rate_1��:

loss_1]�V@���5       ��]�	A�Ӥ���A�*'

learning_rate_1��:

loss_1�xK@EJw�5       ��]�	f"����A�*'

learning_rate_1��:

loss_1A}R@�}�Z5       ��]�	�u ����A�*'

learning_rate_1��:

loss_1:1=@��We5       ��]�	9�����A�*'

learning_rate_1��:

loss_1�G@4��\5       ��]�	? -����A�*'

learning_rate_1��:

loss_1�:@Q�k�5       ��]�	�?C����A�*'

learning_rate_1��:

loss_1�LT@����5       ��]�	�FY����A�*'

learning_rate_1��:

loss_1�H@��5       ��]�	��o����A�*'

learning_rate_1��:

loss_1+�M@�C��5       ��]�	t������A�*'

learning_rate_1��:

loss_1�@O@_���5       ��]�	������A�*'

learning_rate_1��:

loss_1KP@�P�5       ��]�	�������A�*'

learning_rate_1��:

loss_1cK@F�f~5       ��]�	P3ɥ���A�*'

learning_rate_1��:

loss_1P�P@H�1�5       ��]�	bߥ���A�*'

learning_rate_1��:

loss_1J@q�:5       ��]�	�������A�*'

learning_rate_1��:

loss_1tS@����5       ��]�	4����A�*'

learning_rate_1��:

loss_1�D@�G 15       ��]�	�-"����A�*'

learning_rate_1��:

loss_1��@@�!��5       ��]�	@k8����A�*'

learning_rate_1��:

loss_1w�A@ {�5       ��]�	�N����A�*'

learning_rate_1��:

loss_1�I@��5       ��]�	��d����A�*'

learning_rate_1��:

loss_1�gK@R�dL5       ��]�	�{����A�*'

learning_rate_1��:

loss_1U@iL�5       ��]�	�������A�*'

learning_rate_1��:

loss_1p�E@��n�5       ��]�	������A�*'

learning_rate_1��:

loss_1�[G@=w�5       ��]�	A	�����A�*'

learning_rate_1��:

loss_1JP@ĭp�5       ��]�	�զ���A�*'

learning_rate_1��:

loss_1�X@)�*F5       ��]�	������A�*'

learning_rate_1��:

loss_1�&H@E�E�5       ��]�	�����A�*'

learning_rate_1��:

loss_1Tw:@��x]5       ��]�	EI����A�*'

learning_rate_1��:

loss_1��L@�_,5       ��]�	#)/����A�*'

learning_rate_1��:

loss_1�R@��3A5       ��]�	R�F����A�*'

learning_rate_1��:

loss_1#�B@d=��5       ��]�	>�\����A�*'

learning_rate_1��:

loss_1g�H@��5       ��]�	�r����A�*'

learning_rate_1��:

loss_1�N@{U�5       ��]�	�;�����A�*'

learning_rate_1��:

loss_1�I@p[�5       ��]�	������A�*'

learning_rate_1��:

loss_1��J@���5       ��]�	�x�����A�*'

learning_rate_1��:

loss_1;�Y@��-5       ��]�	ģ���A�*'

learning_rate_1��:

loss_1�VL@!�#�5       ��]�	Ű����A�*'

learning_rate_1��:

loss_1�|I@�U�5       ��]�	�������A�*'

learning_rate_1��:

loss_1H;@At�Z5       ��]�	�����A�*'

learning_rate_1��:

loss_1o�O@�=Y5       ��]�	��'����A�*'

learning_rate_1��:

loss_1K|E@���5       ��]�	��=����A�*'

learning_rate_1��:

loss_1��Q@�4��5       ��]�	�9T����A�*'

learning_rate_1��:

loss_1=�P@�w�Y5       ��]�	�oj����A�*'

learning_rate_1��:

loss_1��S@-�{5       ��]�	~������A�*'

learning_rate_1��:

loss_1�^_@e���5       ��]�	8�����A�*'

learning_rate_1��:

loss_1��E@X��g5       ��]�	�������A�*'

learning_rate_1��:

loss_1�;@V�5       ��]�	�Ũ���A�*'

learning_rate_1��:

loss_1!q]@�Q5       ��]�	8�ܨ���A�*'

learning_rate_1��:

loss_1��M@�F*n5       ��]�	�M����A�*'

learning_rate_1��:

loss_1��M@I��5       ��]�	Nl	����A�*'

learning_rate_1��:

loss_1{U@�v�T5       ��]�	6 ����A�*'

learning_rate_1��:

loss_1`�F@j�A~5       ��]�	h�6����A�*'

learning_rate_1��:

loss_1�jB@���	5       ��]�		'P����A�*'

learning_rate_1��:

loss_1�B@�c��5       ��]�	7@f����A�*'

learning_rate_1��:

loss_1��<@o�V�5       ��]�	�@z����A�*'

learning_rate_1��:

loss_1z�e@T��5       ��]�	������A�*'

learning_rate_1��:

loss_1��O@�;ƈ5       ��]�	�������A�*'

learning_rate_1��:

loss_1��G@�ȼH5       ��]�	u�����A�*'

learning_rate_1��:

loss_1�lK@�׈5       ��]�	��ԩ���A�*'

learning_rate_1��:

loss_1��F@�4�M5       ��]�	A����A�*'

learning_rate_1��:

loss_17,H@����5       ��]�	�V����A�*'

learning_rate_1��:

loss_1��;@�C	5       ��]�	C�����A�*'

learning_rate_1��:

loss_1�)Q@JMt5       ��]�	54.����A�*'

learning_rate_1��:

loss_1�\S@���55       ��]�	�aD����A�*'

learning_rate_1��:

loss_1��S@'�E�5       ��]�	l�Z����A�*'

learning_rate_1��:

loss_1�9O@�7bO5       ��]�	bq����A�*'

learning_rate_1��:

loss_1�E@B��5       ��]�	�?�����A�*'

learning_rate_1��:

loss_1��P@҉��5       ��]�	Di�����A�*'

learning_rate_1��:

loss_14F@f ��5       ��]�	�'�����A�*'

learning_rate_1��:

loss_1"zG@�d�5       ��]�	�ʪ���A�*'

learning_rate_1��:

loss_1E�J@L�-�5       ��]�	�c����A�*'

learning_rate_1��:

loss_1NWD@s�W5       ��]�	������A�*'

learning_rate_1��:

loss_1�k?@Άe�5       ��]�	�����A�*'

learning_rate_1��:

loss_1�W@յ!5       ��]�	c#����A�*'

learning_rate_1��:

loss_1�&8@�K�5       ��]�	��9����A�*'

learning_rate_1��:

loss_1��Q@�|5       ��]�	��O����A�*'

learning_rate_1��:

loss_1N|Q@X�y5       ��]�	SFf����A�*'

learning_rate_1��:

loss_1`.J@rpX�5       ��]�	ׂ}����A�*'

learning_rate_1��:

loss_1��>@��X�5       ��]�	1̓����A�*'

learning_rate_1��:

loss_1cL@�h�5       ��]�	Gʩ����A�*'

learning_rate_1��:

loss_1i�M@�T:�5       ��]�	������A�*'

learning_rate_1��:

loss_1�|P@�@�15       ��]�	�Z֫���A�*'

learning_rate_1��:

loss_1�N@��K5       ��]�	
�����A�*'

learning_rate_1��:

loss_1ZG@�czo5       ��]�	������A�*'

learning_rate_1��:

loss_1�M@@�O�5       ��]�	������A�*'

learning_rate_1��:

loss_1 D@gۣ5       ��]�	V0����A�*'

learning_rate_1��:

loss_1%oE@�r-�5       ��]�	�AF����A�*'

learning_rate_1��:

loss_1�]Q@^���5       ��]�	 W\����A�*'

learning_rate_1��:

loss_1�;@�XZ�5       ��]�	��r����A�*'

learning_rate_1��:

loss_1� R@���5       ��]�	Ã�����A�*'

learning_rate_1��:

loss_1�}W@IH��5       ��]�	�ʠ����A�*'

learning_rate_1��:

loss_1@D@t��5       ��]�	�Ӷ����A�*'

learning_rate_1��:

loss_1�J@��8�5       ��]�	fͬ���A�*'

learning_rate_1��:

loss_1�#C@�5       ��]�	������A�*'

learning_rate_1��:

loss_1muF@�R!I5       ��]�	x�����A�*'

learning_rate_1��:

loss_1KO@aLf�5       ��]�	������A�*'

learning_rate_1��:

loss_1�|T@vBs5       ��]�	.'����A�*'

learning_rate_1��:

loss_1 `Q@��5       ��]�	��=����A�*'

learning_rate_1��:

loss_1=�X@�Ǿ�5       ��]�	/T����A�*'

learning_rate_1��:

loss_1y9I@���\5       ��]�	�ej����A�*'

learning_rate_1��:

loss_1�@@Ԑ5!5       ��]�	������A�*'

learning_rate_1��:

loss_1BD@��W�5       ��]�	"�����A�*'

learning_rate_1��:

loss_1_�G@Ӫ%%5       ��]�	�������A�*'

learning_rate_1��:

loss_1oB@��*5       ��]�	̭���A�*'

learning_rate_1�[�:

loss_12N@� 5       ��]�	L�����A�*'

learning_rate_1�[�:

loss_1{�H@��r�5       ��]�	H������A�*'

learning_rate_1�[�:

loss_1�S@M��5       ��]�	s�����A�*'

learning_rate_1�[�:

loss_1�:@�5       ��]�	e�%����A�*'

learning_rate_1�[�:

loss_1��D@�G�5       ��]�	��;����A�*'

learning_rate_1�[�:

loss_1N�S@/��H5       ��]�	��R����A�*'

learning_rate_1�[�:

loss_1�E@��s�5       ��]�	Si����A�*'

learning_rate_1�[�:

loss_1��[@���5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�	J@��75       ��]�	)ϖ����A�*'

learning_rate_1�[�:

loss_1(3F@�_�5       ��]�	{�����A�*'

learning_rate_1�[�:

loss_1ҫN@�(>T5       ��]�	�uî���A�*'

learning_rate_1�[�:

loss_10�J@��5       ��]�	¢ٮ���A�*'

learning_rate_1�[�:

loss_1�e@@�ǿ65       ��]�	}�����A�*'

learning_rate_1�[�:

loss_1dG@^)75       ��]�	�����A�*'

learning_rate_1�[�:

loss_1�6Q@{�uG5       ��]�	�p����A�*'

learning_rate_1�[�:

loss_1}]M@��h�5       ��]�	ܦ2����A�*'

learning_rate_1�[�:

loss_1�E@퐋�5       ��]�	[J����A�*'

learning_rate_1�[�:

loss_1��O@�̃5       ��]�	Z�a����A�*'

learning_rate_1�[�:

loss_1J@�n�5       ��]�	2	y����A�*'

learning_rate_1�[�:

loss_1�F@$j��5       ��]�	�8�����A�*'

learning_rate_1�[�:

loss_1��K@��05       ��]�	*������A�*'

learning_rate_1�[�:

loss_1�cK@�gz�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��W@�L��5       ��]�	
�ԯ���A�*'

learning_rate_1�[�:

loss_1��4@����5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1t*P@n��U5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1HS@�DR�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��F@�ތl5       ��]�	�-����A�*'

learning_rate_1�[�:

loss_1��H@��&5       ��]�	kHC����A�*'

learning_rate_1�[�:

loss_1��I@�0#|5       ��]�	�:Y����A�*'

learning_rate_1�[�:

loss_1t�>@އ��5       ��]�	�ho����A�*'

learning_rate_1�[�:

loss_1LlJ@ή`�5       ��]�	�醰���A�*'

learning_rate_1�[�:

loss_1UvF@�r"5       ��]�	+������A�*'

learning_rate_1�[�:

loss_1�`K@͂�5       ��]�	B޳����A�*'

learning_rate_1�[�:

loss_1?�D@�O5       ��]�	� ʰ���A�*'

learning_rate_1�[�:

loss_1`�O@R;5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�ID@��5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��?@��5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�B@�T�5       ��]�	��%����A�*'

learning_rate_1�[�:

loss_1�k1@�Q� 5       ��]�	�<����A�*'

learning_rate_1�[�:

loss_1V�;@�V�5       ��]�	�`R����A�*'

learning_rate_1�[�:

loss_1�
>@`ޗ5       ��]�	5�h����A�*'

learning_rate_1�[�:

loss_1�I@�H_5       ��]�	f����A�*'

learning_rate_1�[�:

loss_1��>@x���5       ��]�	$w�����A�*'

learning_rate_1�[�:

loss_1��G@�E�Z5       ��]�	�k�����A�*'

learning_rate_1�[�:

loss_1P|N@���5       ��]�	��±���A�*'

learning_rate_1�[�:

loss_1�wA@[�UB5       ��]�	.Hٱ���A�*'

learning_rate_1�[�:

loss_1�D@���5       ��]�	F�����A�*'

learning_rate_1�[�:

loss_1
�M@�F�g5       ��]�	������A�*'

learning_rate_1�[�:

loss_1~K>@���Z5       ��]�	�h����A�*'

learning_rate_1�[�:

loss_1��9@�[�5       ��]�	��2����A�*'

learning_rate_1�[�:

loss_1��K@��5       ��]�	�I����A�*'

learning_rate_1�[�:

loss_1��H@��sU5       ��]�	Ĕ_����A�*'

learning_rate_1�[�:

loss_1�eK@����5       ��]�	@@v����A�*'

learning_rate_1�[�:

loss_1@�T@��R�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�;U@�U)5       ��]�	�"�����A�*'

learning_rate_1�[�:

loss_1BPI@�_�i5       ��]�	�\�����A�*'

learning_rate_1�[�:

loss_15E@0[��5       ��]�	*�ϲ���A�*'

learning_rate_1�[�:

loss_1��M@���5       ��]�	a�����A�*'

learning_rate_1�[�:

loss_1�i@ǬI5       ��]�	q2�����A�*'

learning_rate_1�[�:

loss_1�T<@�g�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��L@�]�I5       ��]�	E�(����A�*'

learning_rate_1�[�:

loss_1�}:@����5       ��]�	F�?����A�*'

learning_rate_1�[�:

loss_16�F@'�KG5       ��]�	r�U����A�*'

learning_rate_1�[�:

loss_1��H@����5       ��]�	Al����A�*'

learning_rate_1�[�:

loss_1V�G@�/�I5       ��]�	:������A�*'

learning_rate_1�[�:

loss_1a\D@�05A5       ��]�	F����A�*'

learning_rate_1�[�:

loss_1TM@��5       ��]�	߄�����A�*'

learning_rate_1�[�:

loss_1��K@M$OX5       ��]�	t�ǳ���A�*'

learning_rate_1�[�:

loss_1��M@��K)5       ��]�	3\޳���A�*'

learning_rate_1�[�:

loss_1�L@��2�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�mX@;��u5       ��]�	�!����A�*'

learning_rate_1�[�:

loss_1{�I@!/5       ��]�	Aa!����A�*'

learning_rate_1�[�:

loss_1,�?@'N�5       ��]�	��7����A�*'

learning_rate_1�[�:

loss_1�)E@��t!5       ��]�	�3N����A�*'

learning_rate_1�[�:

loss_1�6>@�rw/5       ��]�	xhd����A�*'

learning_rate_1�[�:

loss_1�G@@��T5       ��]�	�z����A�*'

learning_rate_1�[�:

loss_1�/P@�-2�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�G@�}
5       ��]�	uV�����A�*'

learning_rate_1�[�:

loss_1�>@4�>5       ��]�	HV�����A�*'

learning_rate_1�[�:

loss_1��T@ ��5       ��]�	lԴ���A�*'

learning_rate_1�[�:

loss_1��N@��r�5       ��]�	�D����A�*'

learning_rate_1�[�:

loss_1�@@�gb}5       ��]�	M� ����A�*'

learning_rate_1�[�:

loss_1��F@-Q��5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�L@7�5       ��]�	"K.����A�*'

learning_rate_1�[�:

loss_1��A@ć�m5       ��]�	�D����A�*'

learning_rate_1�[�:

loss_1�s?@{�5       ��]�	��Z����A�*'

learning_rate_1�[�:

loss_1
wV@�d�5       ��]�	�Cq����A�*'

learning_rate_1�[�:

loss_1�CV@��$�5       ��]�	�|�����A�*'

learning_rate_1�[�:

loss_1�O@'җ%5       ��]�	a������A�*'

learning_rate_1�[�:

loss_1�-J@��}"5       ��]�	#�����A�*'

learning_rate_1�[�:

loss_1j6E@�|g�5       ��]�	�~ʵ���A�*'

learning_rate_1�[�:

loss_14I@b���5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1}�C@�'V5       ��]�	�r�����A�*'

learning_rate_1�[�:

loss_1��P@�o��5       ��]�	C�����A�*'

learning_rate_1�[�:

loss_1�
D@~xz5       ��]�	��#����A�*'

learning_rate_1�[�:

loss_1%�J@ ^� 5       ��]�	�9:����A�*'

learning_rate_1�[�:

loss_1�bM@sI��5       ��]�	��Q����A�*'

learning_rate_1�[�:

loss_1��>@Ch�5       ��]�	�`h����A�*'

learning_rate_1�[�:

loss_1��D@u�^]5       ��]�	A�~����A�*'

learning_rate_1�[�:

loss_1�aM@��5       ��]�	ge�����A�*'

learning_rate_1�[�:

loss_1�&C@�Q�B5       ��]�	j������A�*'

learning_rate_1�[�:

loss_1�C@ڱ05       ��]�	��ö���A�*'

learning_rate_1�[�:

loss_1�I@���n5       ��]�	T�ٶ���A�*'

learning_rate_1�[�:

loss_1��K@)�7�5       ��]�	/����A�*'

learning_rate_1�[�:

loss_1��I@c��5       ��]�	X����A�*'

learning_rate_1�[�:

loss_1�1H@Q��5       ��]�	X^����A�*'

learning_rate_1�[�:

loss_1΍A@y׼�5       ��]�	��3����A�*'

learning_rate_1�[�:

loss_1v#H@F�z�5       ��]�	��I����A�*'

learning_rate_1�[�:

loss_1�(L@g�5       ��]�	ҟ`����A�*'

learning_rate_1�[�:

loss_1>�B@Y�#5       ��]�	X5w����A�*'

learning_rate_1�[�:

loss_1RV@_���5       ��]�	'卷���A�*'

learning_rate_1�[�:

loss_1{DH@�m[�5       ��]�	q�����A�*'

learning_rate_1�[�:

loss_1��G@髙�5       ��]�	mP�����A�*'

learning_rate_1�[�:

loss_1�lO@�cAz5       ��]�	��з���A�*'

learning_rate_1�[�:

loss_1��]@��15       ��]�	�&����A�*'

learning_rate_1�[�:

loss_1�nE@�`5       ��]�	R�����A�*'

learning_rate_1�[�:

loss_1yE@IG�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�s:@�Ҟ05       ��]�	,�*����A�*'

learning_rate_1�[�:

loss_1P!<@�4H�5       ��]�	0"A����A�*'

learning_rate_1�[�:

loss_1*�I@�[�q5       ��]�	�dX����A�*'

learning_rate_1�[�:

loss_1!�G@���n5       ��]�	ۮn����A�*'

learning_rate_1�[�:

loss_1��O@�r{5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�S@���,5       ��]�	%]�����A�*'

learning_rate_1�[�:

loss_1�oR@W{�5       ��]�	Y}�����A�*'

learning_rate_1�[�:

loss_1��A@~�|5       ��]�	{�Ǹ���A�*'

learning_rate_1�[�:

loss_1��D@�*j�5       ��]�	J޸���A�*'

learning_rate_1�[�:

loss_1��C@4��x5       ��]�	b@�����A�*'

learning_rate_1�[�:

loss_1#�R@�T"�5       ��]�		a����A�*'

learning_rate_1�[�:

loss_1[GE@����5       ��]�	��"����A�*'

learning_rate_1�[�:

loss_1_�U@=��M5       ��]�	_k:����A�*'

learning_rate_1�[�:

loss_1I�C@O+��5       ��]�	V	R����A�*'

learning_rate_1�[�:

loss_1��>@=�Q35       ��]�	>h����A�*'

learning_rate_1�[�:

loss_1�Z?@�3?5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1�f9@>��5       ��]�	0s�����A�*'

learning_rate_1�[�:

loss_1�tM@���5       ��]�	R�����A�*'

learning_rate_1�[�:

loss_1�:I@޶�5       ��]�	x�ù���A�*'

learning_rate_1�[�:

loss_1-2@@B�5       ��]�	��ٹ���A�*'

learning_rate_1�[�:

loss_1��]@�5       ��]�	�0����A�*'

learning_rate_1�[�:

loss_1$�@@M�E#5       ��]�	9�����A�*'

learning_rate_1�[�:

loss_1DmH@�a�5       ��]�	2/����A�*'

learning_rate_1�[�:

loss_1y^@*P��5       ��]�	�{3����A�*'

learning_rate_1�[�:

loss_1��7@>�T~5       ��]�	 �I����A�*'

learning_rate_1�[�:

loss_15�M@ ���5       ��]�	
+`����A�*'

learning_rate_1�[�:

loss_1��O@_��b5       ��]�	�v����A�*'

learning_rate_1�[�:

loss_1�KL@?z`�5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1USB@6΀5       ��]�	�ۣ����A�*'

learning_rate_1�[�:

loss_1�mD@�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1b�H@�1��5       ��]�	�iѺ���A�*'

learning_rate_1�[�:

loss_1��M@*v��5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�Y=@8>H	5       ��]�	�������A�*'

learning_rate_1�[�:

loss_14�I@Y �"5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1Q�@@>��5       ��]�	PR*����A�*'

learning_rate_1�[�:

loss_1<�Q@�'<5       ��]�	&�A����A�*'

learning_rate_1�[�:

loss_1�jF@Q���5       ��]�	�0X����A�*'

learning_rate_1�[�:

loss_1vO@�5       ��]�	t�n����A�*'

learning_rate_1�[�:

loss_1N?R@yW�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�!@@�ބ*5       ��]�	e�����A�*'

learning_rate_1�[�:

loss_1�'L@��t55       ��]�	�?�����A�*'

learning_rate_1�[�:

loss_1�	E@C�t5       ��]�	��Ȼ���A�*'

learning_rate_1�[�:

loss_1�TI@��M5       ��]�	�߻���A�*'

learning_rate_1�[�:

loss_1Վ?@�T�5       ��]�	|������A�*'

learning_rate_1�[�:

loss_1��P@i�Y5       ��]�	WF����A�*'

learning_rate_1�[�:

loss_1j�I@�6|5       ��]�	��"����A�*'

learning_rate_1�[�:

loss_1EMJ@�m��5       ��]�	�w9����A�*'

learning_rate_1�[�:

loss_1^�W@qq;�5       ��]�	��O����A�*'

learning_rate_1�[�:

loss_1��_@��p5       ��]�	[$f����A�*'

learning_rate_1�[�:

loss_1iIJ@5�d�5       ��]�	��|����A�*'

learning_rate_1�[�:

loss_1L�I@���(5       ��]�	<������A�*'

learning_rate_1�[�:

loss_1{�8@Lظ5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1lnC@!��5       ��]�	�׿����A�*'

learning_rate_1�[�:

loss_1~tR@⣵�5       ��]�	^jּ���A�*'

learning_rate_1�[�:

loss_1y�@@Y�t55       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�s8@ǜ�5       ��]�	%M����A�*'

learning_rate_1�[�:

loss_1~�@@�%�f5       ��]�	6a����A�*'

learning_rate_1�[�:

loss_1��B@���5       ��]�	�0����A�*'

learning_rate_1�[�:

loss_1#`I@frZ}5       ��]�	�/H����A�*'

learning_rate_1�[�:

loss_1^�@@Fe��5       ��]�	n_����A�*'

learning_rate_1�[�:

loss_1�@@#���5       ��]�	�`u����A�*'

learning_rate_1�[�:

loss_1j G@��wd5       ��]�	�~�����A�*'

learning_rate_1�[�:

loss_1�B@���_5       ��]�	k������A�*'

learning_rate_1�[�:

loss_1�E@w��|5       ��]�	�2�����A�*'

learning_rate_1�[�:

loss_1܁A@�&%�5       ��]�	 ѽ���A�*'

learning_rate_1�[�:

loss_1�07@�� 5       ��]�	�X����A�*'

learning_rate_1�[�:

loss_1�JR@�!+�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��?@Y#5       ��]�	x$����A�*'

learning_rate_1�[�:

loss_1��C@��:�5       ��]�	_*����A�*'

learning_rate_1�[�:

loss_1��;@����5       ��]�	͢@����A�*'

learning_rate_1�[�:

loss_1B�F@�[��5       ��]�	`�V����A�*'

learning_rate_1�[�:

loss_1+UH@Ә�5       ��]�	m����A�*'

learning_rate_1�[�:

loss_1��F@��{5       ��]�	�1�����A�*'

learning_rate_1�[�:

loss_1(�J@���5       ��]�	ܚ����A�*'

learning_rate_1�[�:

loss_1A"I@�4�!5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1�*B@S3(C5       ��]�	��Ǿ���A�*'

learning_rate_1�[�:

loss_1�R@;5       ��]�	O߾���A�*'

learning_rate_1�[�:

loss_1�qC@���"5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�gJ@��,5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1U�;@6�XB5       ��]�	�Q"����A�*'

learning_rate_1�[�:

loss_1*0E@���'5       ��]�	�8����A�*'

learning_rate_1�[�:

loss_1��U@�m]	5       ��]�	f�N����A�*'

learning_rate_1�[�:

loss_1J�Z@b�QZ5       ��]�	��d����A�*'

learning_rate_1�[�:

loss_1C�B@=:�R5       ��]�	*4{����A�*'

learning_rate_1�[�:

loss_1%�>@��^5       ��]�	 ������A�*'

learning_rate_1�[�:

loss_1_�<@<�d5       ��]�	F������A�*'

learning_rate_1�[�:

loss_1}�1@���5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1�VC@5ׅ'5       ��]�	U�׿���A�*'

learning_rate_1�[�:

loss_1aQM@�*5       ��]�	f������A�*'

learning_rate_1�[�:

loss_1�[>@�i�5       ��]�	l�����A�*'

learning_rate_1�[�:

loss_1��L@��&�5       ��]�	�B����A�*'

learning_rate_1�[�:

loss_1D�J@��5�5       ��]�	~0����A�*'

learning_rate_1�[�:

loss_1�$=@ܑ�	5       ��]�	)�F����A�*'

learning_rate_1�[�:

loss_1fL@��xw5       ��]�	��\����A�*'

learning_rate_1�[�:

loss_1иJ@���5       ��]�	:6s����A�*'

learning_rate_1�[�:

loss_1դ;@���M5       ��]�	�u�����A�*'

learning_rate_1�[�:

loss_1a�D@�"ԝ5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�8@S��5       ��]�		I�����A�*'

learning_rate_1�[�:

loss_1�lP@���?5       ��]�	9:�����A�*'

learning_rate_1�[�:

loss_1�0C@y��5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1T=D@�ZƲ5       ��]�	pL�����A�*'

learning_rate_1�[�:

loss_1݆N@q�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1׻L@ƚ5       ��]�	��&����A�*'

learning_rate_1�[�:

loss_1�J@��5       ��]�	z�>����A�*'

learning_rate_1�[�:

loss_1M�D@��[5       ��]�	��U����A�*'

learning_rate_1�[�:

loss_1	�J@y�5       ��]�	w7l����A�*'

learning_rate_1�[�:

loss_1�(W@m��M5       ��]�	ig�����A�*'

learning_rate_1�[�:

loss_1BN@��s5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�QF@��e�5       ��]�	, �����A�*'

learning_rate_1�[�:

loss_1r�L@�5       ��]�	�2�����A�*'

learning_rate_1�[�:

loss_1
�;@��Ri5       ��]�	�5�����A�*'

learning_rate_1�[�:

loss_1�vK@�(ĝ5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1B�M@�]�5       ��]�	(�
����A�*'

learning_rate_1�[�:

loss_1�`.@��}�5       ��]�	�"����A�*'

learning_rate_1�[�:

loss_1�4-@�,e�5       ��]�	Դ8����A�*'

learning_rate_1�[�:

loss_1��F@����5       ��]�	�O����A�*'

learning_rate_1�[�:

loss_1ZLO@�[��5       ��]�	l�e����A�*'

learning_rate_1�[�:

loss_1*�N@�A-�5       ��]�	�z����A�*'

learning_rate_1�[�:

loss_1N�Z@��5       ��]�	�ϐ����A�*'

learning_rate_1�[�:

loss_1\RN@Wh��5       ��]�	������A�*'

learning_rate_1�[�:

loss_1HQ@c���5       ��]�	IN�����A�*'

learning_rate_1�[�:

loss_1ߐQ@���5       ��]�	8������A�*'

learning_rate_1�[�:

loss_1� R@�_C�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��M@��
!5       ��]�	�M ����A�*'

learning_rate_1�[�:

loss_1K@���5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1HlN@[���5       ��]�	�-����A�*'

learning_rate_1�[�:

loss_1&cI@�ySq5       ��]�		�B����A�*'

learning_rate_1�[�:

loss_1�L@�n-5       ��]�	��Z����A�*'

learning_rate_1�[�:

loss_1��9@RI��5       ��]�	�*q����A�*'

learning_rate_1�[�:

loss_1��B@ոŝ5       ��]�	n������A�*'

learning_rate_1�[�:

loss_1|�A@�o�t5       ��]�	]������A�*'

learning_rate_1�[�:

loss_1��F@���x5       ��]�	�^�����A�*'

learning_rate_1�[�:

loss_1(�?@�4f5       ��]�	Pn�����A�*'

learning_rate_1�[�:

loss_1ޝ;@���35       ��]�	�������A�*'

learning_rate_1�[�:

loss_1߾C@�řn5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�tK@;s5       ��]�	]�����A�*'

learning_rate_1�[�:

loss_1=N@���E5       ��]�	�\'����A�*'

learning_rate_1�[�:

loss_1��O@���`5       ��]�	�F=����A�*'

learning_rate_1�[�:

loss_1ON@�6=s5       ��]�	�_S����A�*'

learning_rate_1�[�:

loss_1mR@�u�5       ��]�	yi����A�*'

learning_rate_1�[�:

loss_1Y>W@�95       ��]�	�����A�*'

learning_rate_1�[�:

loss_1��Q@ύx�5       ��]�	eޖ����A�*'

learning_rate_1�[�:

loss_1I�O@R�2)5       ��]�	�r�����A�*'

learning_rate_1�[�:

loss_1S�A@���|5       ��]�	b������A�*'

learning_rate_1�[�:

loss_1�W?@E�y)5       ��]�	������A�*'

learning_rate_1�[�:

loss_1!�@@σ��5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�;@D`�5       ��]�	M����A�*'

learning_rate_1�[�:

loss_1
gR@U ��5       ��]�	L[����A�*'

learning_rate_1�[�:

loss_1�5@j��5       ��]�	ܼ5����A�*'

learning_rate_1�[�:

loss_1�?@Rq��5       ��]�	Y�K����A�*'

learning_rate_1�[�:

loss_1�UF@�=��5       ��]�	\pb����A�*'

learning_rate_1�[�:

loss_1�J@��Q�5       ��]�	>�x����A�*'

learning_rate_1�[�:

loss_1��D@[u��5       ��]�	f�����A�*'

learning_rate_1�[�:

loss_1P1K@tx��5       ��]�	M/�����A�*'

learning_rate_1�[�:

loss_17[I@�!t�5       ��]�	R������A�*'

learning_rate_1�[�:

loss_1K�I@���5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��B@����5       ��]�	:g�����A�*'

learning_rate_1�[�:

loss_1y1G@�*5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1Y�P@�Q��5       ��]�	m�����A�*'

learning_rate_1�[�:

loss_1�
H@Rn&�5       ��]�	X�+����A�*'

learning_rate_1�[�:

loss_1I�5@�B��5       ��]�	��A����A�*'

learning_rate_1�[�:

loss_1��O@�O�p5       ��]�	�=X����A�*'

learning_rate_1�[�:

loss_1�@@j�a�5       ��]�	��o����A�*'

learning_rate_1�[�:

loss_1:@@L�L�5       ��]�	�C�����A�*'

learning_rate_1�[�:

loss_1sbV@�y��5       ��]�	P�����A�*'

learning_rate_1�[�:

loss_1P%C@Ա�5       ��]�	�ҳ����A�*'

learning_rate_1�[�:

loss_1��G@AA?5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��B@Q	F�5       ��]�	4�����A�*'

learning_rate_1�[�:

loss_1![Y@���*5       ��]�	DJ�����A�*'

learning_rate_1�[�:

loss_1	�P@�W�5       ��]�	V�����A�*'

learning_rate_1�[�:

loss_1�U@�0��5       ��]�	�B#����A�*'

learning_rate_1�[�:

loss_1�/E@@-`�5       ��]�	��9����A�*'

learning_rate_1�[�:

loss_1��R@�|q5       ��]�	x�M����A�*'

learning_rate_1�[�:

loss_1H2R@�ǑG5       ��]�	�yd����A�*'

learning_rate_1�[�:

loss_10OP@p>��5       ��]�	I�z����A�*'

learning_rate_1�[�:

loss_1CkB@э�5       ��]�	l
�����A�*'

learning_rate_1�[�:

loss_1�F<@�Z�Y5       ��]�	�!�����A�*'

learning_rate_1�[�:

loss_1�xI@��5       ��]�	�R�����A�*'

learning_rate_1�[�:

loss_1Q�B@q7�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�RE@�/[5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��P@���Z5       ��]�	c����A�*'

learning_rate_1�[�:

loss_1��G@9���5       ��]�	8�����A�*'

learning_rate_1�[�:

loss_1�2L@l�o�5       ��]�	��-����A�*'

learning_rate_1�[�:

loss_1��Q@R1�=5       ��]�	8?D����A�*'

learning_rate_1�[�:

loss_1�FS@�W��5       ��]�	��Z����A�*'

learning_rate_1�[�:

loss_1\BM@��&5       ��]�	��p����A�*'

learning_rate_1�[�:

loss_1(uI@Պz�5       ��]�	x������A�*'

learning_rate_1�[�:

loss_1��E@�m"5       ��]�	1;�����A�*'

learning_rate_1�[�:

loss_1�M@�<$\5       ��]�	�F�����A�*'

learning_rate_1�[�:

loss_1ːN@��pD5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1?:O@�� D5       ��]�	V������A�*'

learning_rate_1�[�:

loss_1�H@f���5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1mJ@�h��5       ��]�	b����A�*'

learning_rate_1�[�:

loss_1�rB@⻃5       ��]�	�%����A�*'

learning_rate_1�[�:

loss_1� K@�'�5       ��]�	)�<����A�*'

learning_rate_1�[�:

loss_1äD@��T5       ��]�	�,S����A�*'

learning_rate_1�[�:

loss_1M�L@�5hm5       ��]�	^�i����A�*'

learning_rate_1�[�:

loss_1��@@K�0�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�=D@繹t5       ��]�	J?�����A�*'

learning_rate_1�[�:

loss_1�xT@X {�5       ��]�	g�����A�*'

learning_rate_1�[�:

loss_1�eF@h��Q5       ��]�	|������A�*'

learning_rate_1�[�:

loss_1�_E@��!�5       ��]�	p������A�*'

learning_rate_1�[�:

loss_1��E@5Mu�5       ��]�	t7�����A�*'

learning_rate_1�[�:

loss_1o+D@�Ö�5       ��]�	$_����A�*'

learning_rate_1�[�:

loss_1�A@����5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��A@�Q5       ��]�	��1����A�*'

learning_rate_1�[�:

loss_1�HH@�Mp�5       ��]�	�~H����A�*'

learning_rate_1�[�:

loss_1�KG@ʡ�5       ��]�	M�^����A�*'

learning_rate_1�[�:

loss_1��@@ �<�5       ��]�	��t����A�*'

learning_rate_1�[�:

loss_1�]d@��]}5       ��]�	?΋����A�*'

learning_rate_1�[�:

loss_1��H@{���5       ��]�	yI�����A�*'

learning_rate_1�[�:

loss_1PC@x$`5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1��G@���l5       ��]�	@������A�*'

learning_rate_1�[�:

loss_1�C;@��l�5       ��]�	� �����A�*'

learning_rate_1�[�:

loss_1!�[@�L��5       ��]�	r:�����A�*'

learning_rate_1�[�:

loss_1��A@�t5       ��]�	\I����A�*'

learning_rate_1�[�:

loss_15vI@Y��C5       ��]�	�(����A�*'

learning_rate_1�[�:

loss_17�D@��`e5       ��]�	��>����A�*'

learning_rate_1�[�:

loss_1�dJ@2��i5       ��]�		V����A�*'

learning_rate_1�[�:

loss_1�ZG@�B��5       ��]�	r,l����A�*'

learning_rate_1�[�:

loss_1�{H@�f�]5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�BN@�h_+5       ��]�	�ǘ����A�*'

learning_rate_1�[�:

loss_1�<@��5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�=B@���5       ��]�	|V�����A�*'

learning_rate_1�[�:

loss_1O@�j��5       ��]�	�Y�����A�*'

learning_rate_1�[�:

loss_1qFF@|��5       ��]�	�O�����A�*'

learning_rate_1�[�:

loss_1�;P@UR5       ��]�	�D����A�*'

learning_rate_1�[�:

loss_1�,<@IJ��5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1�41@m��{5       ��]�	s�3����A�*'

learning_rate_1�[�:

loss_1�QJ@���	5       ��]�	1J����A�*'

learning_rate_1�[�:

loss_1��?@��%5       ��]�	Fa����A�*'

learning_rate_1�[�:

loss_1��F@i�5       ��]�	�nw����A�*'

learning_rate_1�[�:

loss_1 >@>ޭ�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�CE@uf�"5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��G@H[;5       ��]�	<7�����A�*'

learning_rate_1�[�:

loss_1:@���5       ��]�	
F�����A�*'

learning_rate_1�[�:

loss_1ʱI@��[5       ��]�	R������A�*'

learning_rate_1�[�:

loss_1d�B@��+A5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�<@T-�5       ��]�	#�����A�*'

learning_rate_1�[�:

loss_14�J@�3o5       ��]�	XR'����A�*'

learning_rate_1�[�:

loss_15�L@Z�`Y5       ��]�	9�=����A�*'

learning_rate_1�[�:

loss_1+�E@:*{5       ��]�	G�S����A�*'

learning_rate_1�[�:

loss_1rcG@=G�	5       ��]�	_ j����A�*'

learning_rate_1�[�:

loss_1-�J@��k5       ��]�	������A�*'

learning_rate_1�[�:

loss_11O@ �\�5       ��]�	Ȗ����A�*'

learning_rate_1�[�:

loss_1�yG@���5       ��]�	������A�*'

learning_rate_1�[�:

loss_1JkX@�ӫ;5       ��]�	4�����A�*'

learning_rate_1�[�:

loss_1&?P@�MqN5       ��]�	;������A�*'

learning_rate_1�[�:

loss_1
�A@�ˁ5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1r@@��3%5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��H@#6�f5       ��]�	2�����A�*'

learning_rate_1�[�:

loss_1v�R@���5       ��]�	�;3����A�*'

learning_rate_1�[�:

loss_1Q�G@�ŧ�5       ��]�	�dI����A�*'

learning_rate_1�[�:

loss_1�lG@�Jb5       ��]�	��_����A�*'

learning_rate_1�[�:

loss_1
E@�+�5       ��]�	!v����A�*'

learning_rate_1�[�:

loss_1�rD@�u�e5       ��]�	'������A�*'

learning_rate_1�[�:

loss_1�aN@���S5       ��]�	|͢����A�*'

learning_rate_1�[�:

loss_1�B@���5       ��]�	gl�����A�*'

learning_rate_1�[�:

loss_1%�?@lp]�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1ߐA@���-5       ��]�	������A�*'

learning_rate_1�[�:

loss_1'pU@�V;5       ��]�	d������A�*'

learning_rate_1�[�:

loss_1?wT@:�35       ��]�	�����A�*'

learning_rate_1�[�:

loss_1��E@By5       ��]�	�p&����A�*'

learning_rate_1�[�:

loss_1�wB@���5       ��]�	�<����A�*'

learning_rate_1�[�:

loss_1�~8@��5       ��]�	�R����A�*'

learning_rate_1�[�:

loss_1��8@��;5       ��]�	3i����A�*'

learning_rate_1�[�:

loss_1�@@hͣ5       ��]�	�~����A�*'

learning_rate_1�[�:

loss_1�;<@��;[5       ��]�	������A�*'

learning_rate_1�[�:

loss_1>@J�F�5       ��]�	�b�����A�*'

learning_rate_1�[�:

loss_1[dQ@r�h}5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��;@�Z�5       ��]�	�J�����A�*'

learning_rate_1�[�:

loss_1kB@�+�5       ��]�	�>�����A�*'

learning_rate_1�[�:

loss_1aJP@,�5       ��]�	~n����A�*'

learning_rate_1�[�:

loss_1��?@E���5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1��7@(0��5       ��]�	O2����A�*'

learning_rate_1�[�:

loss_1��M@A.��5       ��]�	��H����A�*'

learning_rate_1�[�:

loss_1!�@@Tk5       ��]�	��^����A�*'

learning_rate_1�[�:

loss_1GG@�3z�5       ��]�	�Ou����A�*'

learning_rate_1�[�:

loss_1��N@�b3�5       ��]�	룋����A�*'

learning_rate_1�[�:

loss_1#�T@C�Q*5       ��]�	����A�*'

learning_rate_1�[�:

loss_1��F@�HD�5       ��]�	������A�*'

learning_rate_1�[�:

loss_14�L@���5       ��]�	ڎ�����A�*'

learning_rate_1�[�:

loss_1��>@�iH5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�P@r�&5       ��]�	�1�����A�*'

learning_rate_1�[�:

loss_1=�@@P��5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1rP@�R��5       ��]�	>�&����A�*'

learning_rate_1�[�:

loss_1S@��!�5       ��]�	z�<����A�*'

learning_rate_1�[�:

loss_1qkS@�h�5       ��]�	S����A�*'

learning_rate_1�[�:

loss_1ډD@è�5       ��]�	:j����A�*'

learning_rate_1�[�:

loss_1jB@w�5       ��]�	c������A�*'

learning_rate_1�[�:

loss_1�<@�+I5       ��]�	ܖ����A�*'

learning_rate_1�[�:

loss_1jhN@�Z&�5       ��]�	������A�*'

learning_rate_1�[�:

loss_12wB@���B5       ��]�	�=�����A�*'

learning_rate_1�[�:

loss_1��R@I�׉5       ��]�	c������A�*'

learning_rate_1�[�:

loss_1�YB@����5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1��V@6a�5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1�TE@��P�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1;�D@(ƴ�5       ��]�	y,4����A�*'

learning_rate_1�[�:

loss_1y�L@և�l5       ��]�	��J����A�*'

learning_rate_1�[�:

loss_1=WH@D��z5       ��]�	��b����A�*'

learning_rate_1�[�:

loss_1�>@܄~	5       ��]�	SZy����A�*'

learning_rate_1�[�:

loss_1СO@���5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1�@@3h�:5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��D@I��g5       ��]�	;ϼ����A�*'

learning_rate_1�[�:

loss_1r�C@���5       ��]�	L5�����A�*'

learning_rate_1�[�:

loss_1�Z7@�4�5       ��]�	w�����A�*'

learning_rate_1�[�:

loss_1��@@ɛN�5       ��]�	x������A�*'

learning_rate_1�[�:

loss_1o�@@�ҵ�5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1��A@§�5       ��]�	�0.����A�*'

learning_rate_1�[�:

loss_1d�2@�ǃ�5       ��]�	`D����A�*'

learning_rate_1�[�:

loss_1aL@�'�5       ��]�	@�Z����A�*'

learning_rate_1�[�:

loss_1:�B@O���5       ��]�	Liq����A�*'

learning_rate_1�[�:

loss_1��L@e�Ӧ5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��N@�:�15       ��]�	�F�����A�*'

learning_rate_1�[�:

loss_1WT@x�5       ��]�	Ä�����A�*'

learning_rate_1�[�:

loss_1uyA@����5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�G@�sy�5       ��]�	EI�����A�*'

learning_rate_1�[�:

loss_1��H@�+"�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1j�4@d�X5       ��]�	�d����A�*'

learning_rate_1�[�:

loss_1՜R@�H�5       ��]�	��%����A�*'

learning_rate_1�[�:

loss_1��F@���5       ��]�	V�;����A�*'

learning_rate_1�[�:

loss_1��E@���?5       ��]�	I�R����A�*'

learning_rate_1�[�:

loss_1��L@ed'5       ��]�	�0i����A�*'

learning_rate_1�[�:

loss_1cB@�V �5       ��]�	mr����A�*'

learning_rate_1�[�:

loss_1w�L@�Q�K5       ��]�	M�����A�*'

learning_rate_1�[�:

loss_1�(:@	��5       ��]�	vS�����A�*'

learning_rate_1�[�:

loss_1nI@���J5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��I@��)/5       ��]�	Q������A�*'

learning_rate_1�[�:

loss_1KF@@�M5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1B�E@��:�5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1�!B@����5       ��]�	K�����A�*'

learning_rate_1�[�:

loss_1UlM@��qS5       ��]�	�c2����A�*'

learning_rate_1�[�:

loss_1X�M@Pd�5       ��]�	�H����A�*'

learning_rate_1�[�:

loss_1v�V@h��5       ��]�	��_����A�*'

learning_rate_1�[�:

loss_1Bp=@k"{5       ��]�	�hv����A�*'

learning_rate_1�[�:

loss_1�lI@_[n�5       ��]�	�w�����A�*'

learning_rate_1�[�:

loss_177<@���5       ��]�	=������A�*'

learning_rate_1�[�:

loss_1vH@XB_S5       ��]�	>�����A�*'

learning_rate_1�[�:

loss_11z<@-yx�5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�SQ@&Y�5       ��]�	�_�����A�*'

learning_rate_1�[�:

loss_1wlJ@�� r5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1SG@b�5       ��]�	C�����A�*'

learning_rate_1�[�:

loss_1�R@h�5       ��]�	p9*����A�*'

learning_rate_1�[�:

loss_1�P@1@��5       ��]�	O�@����A�*'

learning_rate_1�[�:

loss_1�P@�vEQ5       ��]�	+�V����A�*'

learning_rate_1�[�:

loss_1O@�!�5       ��]�	��l����A�*'

learning_rate_1�[�:

loss_1z�B@���5       ��]�	(Є����A�*'

learning_rate_1�[�:

loss_1o�8@�?�5       ��]�	�F�����A�*'

learning_rate_1�[�:

loss_1�s5@����5       ��]�	�\�����A�*'

learning_rate_1�[�:

loss_1��N@��5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��L@��v5       ��]�	�X�����A�*'

learning_rate_1�[�:

loss_1kTN@�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1+AQ@���5       ��]�	������A�*'

learning_rate_1�[�:

loss_1��R@��
�5       ��]�	�!����A�*'

learning_rate_1�[�:

loss_14
B@\�^5       ��]�	1�7����A�*'

learning_rate_1�[�:

loss_1�O>@����5       ��]�	EN����A�*'

learning_rate_1�[�:

loss_1N@��!v5       ��]�	�Md����A�*'

learning_rate_1�[�:

loss_1�PJ@��I5       ��]�	�hz����A�*'

learning_rate_1�[�:

loss_1�9Q@�V�J5       ��]�	������A�*'

learning_rate_1�[�:

loss_1*�B@+�5       ��]�	[�����A�*'

learning_rate_1�[�:

loss_1hH@�Fb5       ��]�	������A�*'

learning_rate_1�[�:

loss_1�{M@���X5       ��]�	{�����A�*'

learning_rate_1�[�:

loss_1�;@F;_5       ��]�	�t�����A�*'

learning_rate_1�[�:

loss_1^�O@�0i5       ��]�	~� ����A�*'

learning_rate_1�[�:

loss_1�xI@�z�5       ��]�	�#����A�*'

learning_rate_1�[�:

loss_1;]F@�7E5       ��]�	X�.����A�*'

learning_rate_1�[�:

loss_11�A@��Q>5       ��]�	�E����A�*'

learning_rate_1�[�:

loss_1�-C@�V"]5       ��]�	��Z����A�*'

learning_rate_1�[�:

loss_1��C@���L5       ��]�	^�r����A�*'

learning_rate_1�[�:

loss_1F�:@���5       ��]�	�W�����A�*'

learning_rate_1�[�:

loss_1��G@_�q�5       ��]�	M������A�*'

learning_rate_1�[�:

loss_1��F@s��5       ��]�	0ٶ����A�*'

learning_rate_1�[�:

loss_1BP@$�A5       ��]�	C������A�*'

learning_rate_1�[�:

loss_15A@X �q5       ��]�	�%�����A�*'

learning_rate_1�[�:

loss_1��K@���5       ��]�	�;�����A�*'

learning_rate_1�[�:

loss_1�J@�Q<5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1��=@��8l5       ��]�	�='����A�*'

learning_rate_1�[�:

loss_1��?@���5       ��]�	�]=����A�*'

learning_rate_1�[�:

loss_1*K@��Q5       ��]�	9�S����A�*'

learning_rate_1�[�:

loss_1��4@o��5       ��]�	�j����A�*'

learning_rate_1�[�:

loss_1�nD@B��|5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1}�H@]vX}5       ��]�	O�����A�*'

learning_rate_1�[�:

loss_1��P@_���5       ��]�	�&�����A�*'

learning_rate_1�[�:

loss_1�[I@ԕ5       ��]�	ۀ�����A�*'

learning_rate_1�[�:

loss_1��5@t��o5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1ZeO@�Z�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1A@����5       ��]�	Y4����A�*'

learning_rate_1�[�:

loss_1>�M@ƨ��5       ��]�	������A�*'

learning_rate_1�[�:

loss_1&�L@ ��5       ��]�	�2����A�*'

learning_rate_1�[�:

loss_1�3W@�rI5       ��]�	�RI����A�*'

learning_rate_1�[�:

loss_1�4I@i��5       ��]�	�_����A�*'

learning_rate_1�[�:

loss_1},C@� 55       ��]�	��v����A�*'

learning_rate_1�[�:

loss_1��<@���o5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1aG@p�|�5       ��]�	�2�����A�*'

learning_rate_1�[�:

loss_1TX@s��r5       ��]�	�n�����A�*'

learning_rate_1�[�:

loss_1�O@d,�5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1(AD@L�]5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1��@@�k��5       ��]�	7u�����A�*'

learning_rate_1�[�:

loss_1�G7@�{I5       ��]�	T4����A�*'

learning_rate_1�[�:

loss_1QN7@bZ5       ��]�	s*����A�*'

learning_rate_1�[�:

loss_1�IH@
���5       ��]�	yA����A�*'

learning_rate_1�[�:

loss_1GqP@q g�5       ��]�	��W����A�*'

learning_rate_1�[�:

loss_1�hH@��D5       ��]�	F2n����A�*'

learning_rate_1�[�:

loss_1��>@EEJ5       ��]�	�>�����A�*'

learning_rate_1�[�:

loss_1sF@~�]5       ��]�	�������A�*'

learning_rate_1�[�:

loss_1g�E@ Tz5       ��]�	$�����A�*'

learning_rate_1�[�:

loss_1��C@)��5       ��]�	�����A�*'

learning_rate_1�[�:

loss_1ZzN@��5       ��]�	TU�����A�*'

learning_rate_1�[�:

loss_1��9@+M�5       ��]�	Bj�����A�*'

learning_rate_1�[�:

loss_1�?U@���t5       ��]�	��
����A�*'

learning_rate_1�[�:

loss_1�`D@�2=5       ��]�	?4!����A�*'

learning_rate_1�[�:

loss_1�QM@��]k5       ��]�	?�7����A�*'

learning_rate_1�[�:

loss_1�iI@,��5       ��]�	��M����A�*'

learning_rate_1�[�:

loss_1�P@s�@5       ��]�	��f����A�*'

learning_rate_1�[�:

loss_1W�3@Ɨ�!5       ��]�	��|����A�*'

learning_rate_1�[�:

loss_1�<@�}X�5       ��]�	*�����A�*'

learning_rate_1�[�:

loss_1�H@I���5       ��]�	�E�����A�*'

learning_rate_1�[�:

loss_1}�N@a���5       ��]�	�-�����A�*'

learning_rate_1�[�:

loss_1��=@a�sx5       ��]�	JT�����A�*'

learning_rate_1�[�:

loss_1S8G@D��x5       ��]�	_U�����A�*'

learning_rate_1�[�:

loss_1�E@(]>5       ��]�	�k����A�*'

learning_rate_1�[�:

loss_1�[E@��$5       ��]�	Ì����A�*'

learning_rate_1�[�:

loss_1�fM@湐x5       ��]�	��-����A�*'

learning_rate_1�[�:

loss_1j�L@��n5       ��]�	FK����A�*'

learning_rate_1���:

loss_1y�;@w#�C5       ��]�	�Ia����A�*'

learning_rate_1���:

loss_1&I@�gܗ5       ��]�	��w����A�*'

learning_rate_1���:

loss_1��>@ly��5       ��]�	�3�����A�*'

learning_rate_1���:

loss_1m�=@I�t�5       ��]�	Od�����A�*'

learning_rate_1���:

loss_1ɨP@W��5       ��]�	}�����A�*'

learning_rate_1���:

loss_1�F@�G��5       ��]�	�y�����A�*'

learning_rate_1���:

loss_1��B@���q5       ��]�	�f�����A�*'

learning_rate_1���:

loss_1K�A@�:(j5       ��]�	C������A�*'

learning_rate_1���:

loss_1#<@R�R5       ��]�	�6����A�*'

learning_rate_1���:

loss_1��>@��J5       ��]�	�,����A�*'

learning_rate_1���:

loss_1�ND@ìH�5       ��]�	 �B����A�*'

learning_rate_1���:

loss_1��J@�ef5       ��]�	��X����A�*'

learning_rate_1���:

loss_1j�N@d1*5       ��]�	E(o����A�*'

learning_rate_1���:

loss_1ͯK@���5       ��]�	�������A�*'

learning_rate_1���:

loss_1��F@`�5       ��]�	������A�*'

learning_rate_1���:

loss_1�?@g���5       ��]�	C0�����A�*'

learning_rate_1���:

loss_1�U@"�<�5       ��]�	�a�����A�*'

learning_rate_1���:

loss_1�JB@^��t5       ��]�	٨�����A�*'

learning_rate_1���:

loss_1��B@�Wf�5       ��]�	������A�*'

learning_rate_1���:

loss_1�i5@c��5       ��]�	8����A�*'

learning_rate_1���:

loss_1��G@/���5       ��]�	�["����A�*'

learning_rate_1���:

loss_1�M:@e��5       ��]�	>�8����A�*'

learning_rate_1���:

loss_1��N@>��5       ��]�	�
Q����A�*'

learning_rate_1���:

loss_1�5<@�i¶5       ��]�	uIg����A�*'

learning_rate_1���:

loss_1dWR@�/�5       ��]�	��}����A�*'

learning_rate_1���:

loss_1RoL@�t�85       ��]�	vߓ����A�*'

learning_rate_1���:

loss_1��L@h�@L5       ��]�	������A�*'

learning_rate_1���:

loss_1&I@>�C�5       ��]�	u@�����A�*'

learning_rate_1���:

loss_17�M@�s ]5       ��]�	g]�����A�*'

learning_rate_1���:

loss_1��F@qy�5       ��]�	�������A�*'

learning_rate_1���:

loss_1��O@�S5       ��]�	������A�*'

learning_rate_1���:

loss_1n�F@��y5       ��]�	�p����A�*'

learning_rate_1���:

loss_1��A@���5       ��]�	��/����A�*'

learning_rate_1���:

loss_1�=K@'�5       ��]�	�F����A�*'

learning_rate_1���:

loss_104W@�T5       ��]�	�]����A�*'

learning_rate_1���:

loss_1$�?@�w��5       ��]�	nVs����A�*'

learning_rate_1���:

loss_1��B@ؓ�5       ��]�	֛�����A�*'

learning_rate_1���:

loss_1
�D@�X5       ��]�	Q������A�*'

learning_rate_1���:

loss_1J�@@p��5       ��]�	Ĺ�����A�*'

learning_rate_1���:

loss_1tb@@A]u5       ��]�	�������A�*'

learning_rate_1���:

loss_1�d;@$�5       ��]�	)�����A�*'

learning_rate_1���:

loss_1��=@��5T5       ��]�	�2�����A�*'

learning_rate_1���:

loss_1	�@@3�$	5       ��]�	�*����A�*'

learning_rate_1���:

loss_1%�O@���E5       ��]�	!�)����A�*'

learning_rate_1���:

loss_1�K@:��5       ��]�	T�?����A�*'

learning_rate_1���:

loss_1,J@���5       ��]�	�4V����A�*'

learning_rate_1���:

loss_16@?@�+$/5       ��]�	y�l����A�*'

learning_rate_1���:

loss_1_G@��>�5       ��]�	 �����A�*'

learning_rate_1���:

loss_1��B@[�ġ5       ��]�	�$�����A�*'

learning_rate_1���:

loss_1�?@qT�5       ��]�	ٱ����A�*'

learning_rate_1���:

loss_1��=@����5       ��]�	�9�����A�*'

learning_rate_1���:

loss_1��5@��O�5       ��]�	�s�����A�*'

learning_rate_1���:

loss_1]YO@�˩�5       ��]�	������A�*'

learning_rate_1���:

loss_1��D@���5       ��]�	|����A�*'

learning_rate_1���:

loss_1��?@j�aR5       ��]�	-�"����A�*'

learning_rate_1���:

loss_1m�?@)�`5       ��]�	�+9����A�*'

learning_rate_1���:

loss_1x<@@~N	5       ��]�	��O����A�*'

learning_rate_1���:

loss_1@�A@<ig�5       ��]�	{4f����A�*'

learning_rate_1���:

loss_1�fO@�վe5       ��]�	�"|����A�*'

learning_rate_1���:

loss_1�@@Y�\5       ��]�	������A�*'

learning_rate_1���:

loss_1��7@Ø��5       ��]�	Q������A�*'

learning_rate_1���:

loss_1Hq;@D�p25       ��]�	_ƾ����A�*'

learning_rate_1���:

loss_1ZWO@%�;�5       ��]�	5������A�*'

learning_rate_1���:

loss_1KQE@Ӊ�5       ��]�	x������A�*'

learning_rate_1���:

loss_1�C@�u�	5       ��]�	������A�*'

learning_rate_1���:

loss_1D�=@PL�L5       ��]�	C����A�*'

learning_rate_1���:

loss_1�\?@�3$�5       ��]�	W0����A�*'

learning_rate_1���:

loss_1aE@�_�5       ��]�	��F����A�*'

learning_rate_1���:

loss_1�FW@�,�.5       ��]�	W�^����A�*'

learning_rate_1���:

loss_1j�8@<(��5       ��]�	Wu����A�*'

learning_rate_1���:

loss_1��I@χe�5       ��]�	�N�����A�*'

learning_rate_1���:

loss_1��K@7D�5       ��]�	�m�����A�*'

learning_rate_1���:

loss_1�>@L�
5       ��]�	�������A�*'

learning_rate_1���:

loss_1��H@g}a�5       ��]�	� �����A�*'

learning_rate_1���:

loss_1]9@��]5       ��]�	�������A�*'

learning_rate_1���:

loss_1þ?@,FD5       ��]�	�������A�*'

learning_rate_1���:

loss_1;�^@��NL5       ��]�	�T����A�*'

learning_rate_1���:

loss_14�M@&��c5       ��]�	^*����A�*'

learning_rate_1���:

loss_1ɬ=@�RFB5       ��]�	�@����A�*'

learning_rate_1���:

loss_1(k=@��U=5       ��]�	��V����A�*'

learning_rate_1���:

loss_1B@�(M5       ��]�	Zm����A�*'

learning_rate_1���:

loss_1D�>@H�wH5       ��]�	�������A�*'

learning_rate_1���:

loss_1
�F@N��5       ��]�	Eڙ����A�*'

learning_rate_1���:

loss_1I@�;5       ��]�	Q�����A�*'

learning_rate_1���:

loss_1_�G@��R�5       ��]�	�h�����A�*'

learning_rate_1���:

loss_1ps=@�at5       ��]�	�������A�*'

learning_rate_1���:

loss_1-|>@B��5       ��]�	�W�����A�*'

learning_rate_1���:

loss_1�B@d���5       ��]�	��	����A�*'

learning_rate_1���:

loss_1X�D@ae}�5       ��]�	������A�*'

learning_rate_1���:

loss_1�"J@T�Z+5       ��]�	�Z6����A�*'

learning_rate_1���:

loss_1�L@��B�5       ��]�	�zL����A�*'

learning_rate_1���:

loss_14EW@�uM15       ��]�	��b����A�*'

learning_rate_1���:

loss_1�kI@؆�5       ��]�	4y����A�*'

learning_rate_1���:

loss_1״N@۬S5       ��]�	*Q�����A�*'

learning_rate_1���:

loss_1��J@m!��5       ��]�	�̥����A�*'

learning_rate_1���:

loss_1��:@܏��5       ��]�	�����A�*'

learning_rate_1���:

loss_1ki8@��5       ��]�	�-�����A�*'

learning_rate_1���:

loss_1�xC@L�d�5       ��]�	������A�*'

learning_rate_1���:

loss_1rLQ@p��5       ��]�	�����A�*'

learning_rate_1���:

loss_10�=@���5       ��]�	�����A�*'

learning_rate_1���:

loss_1%6@�x5       ��]�	�m/����A�*'

learning_rate_1���:

loss_1�>@B4C5       ��]�	�F����A�*'

learning_rate_1���:

loss_1p�A@��p5       ��]�	K�\����A�*'

learning_rate_1���:

loss_1��J@U��5       ��]�	�
s����A�*'

learning_rate_1���:

loss_1�A@���^5       ��]�	l�����A�*'

learning_rate_1���:

loss_1��B@�%.75       ��]�	@6�����A�*'

learning_rate_1���:

loss_1�LV@�W5       ��]�	$������A�*'

learning_rate_1���:

loss_1$E@�1��5       ��]�	4������A�*'

learning_rate_1���:

loss_17/Q@/ӭ�5       ��]�	~������A�*'

learning_rate_1���:

loss_1J@խ�5       ��]�	�������A�*'

learning_rate_1���:

loss_1��H@��5       ��]�	U�����A�*'

learning_rate_1���:

loss_1�<@.���5       ��]�	��%����A�*'

learning_rate_1���:

loss_1$�G@w��5       ��]�	5�=����A�*'

learning_rate_1���:

loss_17�9@�!�5       ��]�	2�S����A�*'

learning_rate_1���:

loss_1^�C@�
05       ��]�	�j����A�*'

learning_rate_1���:

loss_1��V@�5�c5       ��]�	������A�*'

learning_rate_1���:

loss_1SDI@�k*5       ��]�	&������A�*'

learning_rate_1���:

loss_1CF@p�g!5       ��]�	�K�����A�*'

learning_rate_1���:

loss_1U�T@�i�~5       ��]�	�s�����A�*'

learning_rate_1���:

loss_1ߕ;@�g��5       ��]�	������A�*'

learning_rate_1���:

loss_1�{E@�<��5       ��]�	������A�*'

learning_rate_1���:

loss_1�F@{ݚ5       ��]�	�+����A�*'

learning_rate_1���:

loss_1�H@���S5       ��]�	|����A�*'

learning_rate_1���:

loss_1J�O@O�R!5       ��]�	��2����A�*'

learning_rate_1���:

loss_1N6@@eN�s5       ��]�	�I����A�*'

learning_rate_1���:

loss_1�<@�yT(5       ��]�	|�_����A�*'

learning_rate_1���:

loss_1�D@z�\r5       ��]�	�v����A�*'

learning_rate_1���:

loss_1I@7~55       ��]�	ό����A�*'

learning_rate_1���:

loss_1�F@3��85       ��]�	ǣ�����A�*'

learning_rate_1���:

loss_1�<@� N5       ��]�	�������A�*'

learning_rate_1���:

loss_1��P@ZJ�o5       ��]�	B������A�*'

learning_rate_1���:

loss_1
�D@W�45       ��]�	�*�����A�*'

learning_rate_1���:

loss_1�>@�>5       ��]�	�% ����A�*'

learning_rate_1���:

loss_1�;H@��OD5       ��]�	������A�*'

learning_rate_1���:

loss_1�S?@�f5       ��]�	��,����A�*'

learning_rate_1���:

loss_1\@=@\J�;5       ��]�	�'C����A�*'

learning_rate_1���:

loss_1�WJ@t\5       ��]�	��Y����A�*'

learning_rate_1���:

loss_1ޯ9@���5       ��]�	p����A�*'

learning_rate_1���:

loss_1�v9@�<5       ��]�	������A�*'

learning_rate_1���:

loss_1��>@\!>b5       ��]�	�<�����A�*'

learning_rate_1���:

loss_1�H@1���5       ��]�	�`�����A�*'

learning_rate_1���:

loss_1k!B@����5       ��]�	c}�����A�*'

learning_rate_1���:

loss_1��C@��5       ��]�	������A�*'

learning_rate_1���:

loss_1��C@�XlL5       ��]�	�������A�*'

learning_rate_1���:

loss_1X�H@� K,5       ��]�	�����A�*'

learning_rate_1���:

loss_1�8@�ل&5       ��]�	�"����A�*'

learning_rate_1���:

loss_1�r>@���Z5       ��]�	��:����A�*'

learning_rate_1���:

loss_1��=@��&�5       ��]�	 IQ����A�*'

learning_rate_1���:

loss_17?@�t�5       ��]�	��h����A�*'

learning_rate_1���:

loss_1�~5@{�5       ��]�	iQ�����A�*'

learning_rate_1���:

loss_1��B@��v�5       ��]�	�������A�*'

learning_rate_1���:

loss_1f-N@@~9�5       ��]�	;߬����A�*'

learning_rate_1���:

loss_1�H@�X�5       ��]�	Y������A�*'

learning_rate_1���:

loss_1�GF@l��h5       ��]�	������A�*'

learning_rate_1���:

loss_1.�6@"��E5       ��]�	�������A�*'

learning_rate_1���:

loss_1t�=@�6Q�5       ��]�	)�����A�*'

learning_rate_1���:

loss_1�;Y@o���5       ��]�	M�����A�*'

learning_rate_1���:

loss_1ڣA@�y�5       ��]�	�|3����A�*'

learning_rate_1���:

loss_1/�F@��(5       ��]�	F�I����A�*'

learning_rate_1���:

loss_1z�L@�T_C5       ��]�	/&a����A�*'

learning_rate_1���:

loss_1F�U@��5       ��]�	t$x����A�*'

learning_rate_1���:

loss_1�E@�>�(5       ��]�	�o�����A�*'

learning_rate_1���:

loss_1/�N@[f�5       ��]�	)������A�*'

learning_rate_1���:

loss_1�wQ@Nc�5       ��]�	�1�����A�*'

learning_rate_1���:

loss_1�:C@��&p5       ��]�	n������A�*'

learning_rate_1���:

loss_1ڽQ@�e�5       ��]�	A�����A�*'

learning_rate_1���:

loss_1�I>@=�}5       ��]�	������A�*'

learning_rate_1���:

loss_1o�C@�c�5       ��]�	"����A�*'

learning_rate_1���:

loss_1ߍZ@z�%%5       ��]�	t�,����A�*'

learning_rate_1���:

loss_1�;@
#ߏ5       ��]�	��B����A�*'

learning_rate_1���:

loss_1JE@1x�5       ��]�	�nY����A�*'

learning_rate_1���:

loss_1d�8@3��5       ��]�	��o����A�*'

learning_rate_1���:

loss_1�H@-���5       ��]�	������A�*'

learning_rate_1���:

loss_1�AM@w��^5       ��]�	������A�*'

learning_rate_1���:

loss_1�B@�!x�5       ��]�	�7�����A�*'

learning_rate_1���:

loss_1
�=@�a
5       ��]�	0�����A�*'

learning_rate_1���:

loss_16n7@���+5       ��]�	�������A�*'

learning_rate_1���:

loss_1�_S@���5       ��]�	�������A�*'

learning_rate_1���:

loss_1K�Q@��5       ��]�	4�����A�*'

learning_rate_1���:

loss_1{>@�h�5       ��]�	�#����A�*'

learning_rate_1���:

loss_1Q�R@����5       ��]�	�O9����A�*'

learning_rate_1���:

loss_1��C@�Y��5       ��]�	8�O����A�*'

learning_rate_1���:

loss_1�(F@��5�5       ��]�	E�e����A�*'

learning_rate_1���:

loss_1
7B@��=�5       ��]�	�A|����A�*'

learning_rate_1���:

loss_1��P@��5       ��]�	�������A�*'

learning_rate_1���:

loss_1�K=@�a�5       ��]�	:!�����A�*'

learning_rate_1���:

loss_1^�D@�j��5       ��]�	Y������A�*'

learning_rate_1���:

loss_1��J@ř��5       ��]�	Ѻ�����A�*'

learning_rate_1���:

loss_1�M@��5       ��]�	�!�����A�*'

learning_rate_1���:

loss_1K�P@�8�5       ��]�	ą����A�*'

learning_rate_1���:

loss_1ݻF@)$^�5       ��]�	A�����A�*'

learning_rate_1���:

loss_1x"C@!#>�5       ��]�	�C0����A�*'

learning_rate_1���:

loss_1B�J@D��5       ��]�	8#D����A�*'

learning_rate_1���:

loss_1�^@!z�45       ��]�	�'\����A�*'

learning_rate_1���:

loss_1��I@(`�5       ��]�	lyq����A�*'

learning_rate_1���:

loss_1�*F@jG25       ��]�	-Շ����A�*'

learning_rate_1���:

loss_1DG@�m9-5       ��]�	c�����A�*'

learning_rate_1���:

loss_1�?@�Z�5       ��]�	�1�����A�*'

learning_rate_1���:

loss_1\�R@�=�5       ��]�	ol�����A�*'

learning_rate_1���:

loss_1�P@�iI	5       ��]�	�������A�*'

learning_rate_1���:

loss_1�FF@��w�5       ��]�	#4�����A�*'

learning_rate_1���:

loss_1�=@Z�J5       ��]�	�o����A�*'

learning_rate_1���:

loss_1G�O@��|5       ��]�	(�&����A�*'

learning_rate_1���:

loss_1�~9@I��5       ��]�	"�<����A�*'

learning_rate_1���:

loss_1eI@ir.�5       ��]�	 �P����A�*'

learning_rate_1���:

loss_1�.[@�F�5       ��]�	pyg����A�*'

learning_rate_1���:

loss_1�i0@7d֏5       ��]�	o�}����A�*'

learning_rate_1���:

loss_1N>7@F���5       ��]�	�������A�*'

learning_rate_1���:

loss_1'PE@��N5       ��]�	�z�����A�*'

learning_rate_1���:

loss_1?L@R�Oq5       ��]�	ӿ����A�*'

learning_rate_1���:

loss_1O�;@��5       ��]�	=�����A�*'

learning_rate_1���:

loss_1�D@o�9�5       ��]�	�������A�*'

learning_rate_1���:

loss_1��A@pG�N5       ��]�	������A�*'

learning_rate_1���:

loss_1��S@�lW5       ��]�	U)����A�*'

learning_rate_1���:

loss_1?�R@��5       ��]�	z1����A�*'

learning_rate_1���:

loss_1��B@	
ޅ5       ��]�	E�G����A�*'

learning_rate_1���:

loss_1q�O@�v�5       ��]�	��^����A�*'

learning_rate_1���:

loss_1�>@�H.5       ��]�	��t����A�*'

learning_rate_1���:

loss_1v�=@�B�J5       ��]�	G�����A�*'

learning_rate_1���:

loss_1SF@w4F5       ��]�	pW�����A�*'

learning_rate_1���:

loss_1aRN@H��`5       ��]�	<������A�*'

learning_rate_1���:

loss_1eC@��ɒ5       ��]�	������A�*'

learning_rate_1���:

loss_1zB@���O5       ��]�	�B�����A�*'

learning_rate_1���:

loss_1nD@�]q�5       ��]�	������A�*'

learning_rate_1���:

loss_1��K@�(�5       ��]�	������A�*'

learning_rate_1���:

loss_1��E@D!�I5       ��]�	��'����A�*'

learning_rate_1���:

loss_13C=@��i"5       ��]�		F>����A�*'

learning_rate_1���:

loss_1ԃL@I��c5       ��]�	��T����A�*'

learning_rate_1���:

loss_1ĳL@�.�5       ��]�	JYk����A�*'

learning_rate_1���:

loss_1��E@H�5       ��]�	H������A�*'

learning_rate_1���:

loss_1;�H@�� �5       ��]�	+������A�*'

learning_rate_1���:

loss_1��I@��#�5       ��]�	�Ӯ����A�*'

learning_rate_1���:

loss_1��I@�\�5       ��]�	�V�����A�*'

learning_rate_1���:

loss_1��?@��0�5       ��]�	x^�����A�*'

learning_rate_1���:

loss_1Q�<@-�?5       ��]�	(�����A�*'

learning_rate_1���:

loss_1~#R@[*5       ��]�	;a����A�*'

learning_rate_1���:

loss_1�KD@��[5       ��]�	��!����A�*'

learning_rate_1���:

loss_1C�@@�0��5       ��]�	�8����A�*'

learning_rate_1���:

loss_1�I@�3O5       ��]�	�IO����A�*'

learning_rate_1���:

loss_1=�H@��Q�5       ��]�	ze����A�*'

learning_rate_1���:

loss_1��9@��s5       ��]�	��{����A�*'

learning_rate_1���:

loss_1�vA@��b75       ��]�	����A�*'

learning_rate_1���:

loss_1�J@C�b�5       ��]�	Ӡ�����A�*'

learning_rate_1���:

loss_1G^6@�o(X5       ��]�	3۾����A�*'

learning_rate_1���:

loss_1<s?@N1$�5       ��]�	�,�����A�*'

learning_rate_1���:

loss_1�R@��35       ��]�	�������A�*'

learning_rate_1���:

loss_1VA@	�5       ��]�	3�����A�*'

learning_rate_1���:

loss_1,6K@=.�*5       ��]�	�4����A�*'

learning_rate_1���:

loss_1R�T@vgD�5       ��]�	|o.����A�*'

learning_rate_1���:

loss_1�J@����5       ��]�	ĹD����A�*'

learning_rate_1���:

loss_1	AJ@�3�e5       ��]�	N,[����A�*'

learning_rate_1���:

loss_1MP@���5       ��]�	@�q����A�*'

learning_rate_1���:

loss_1�?@�E'5       ��]�	�������A�*'

learning_rate_1���:

loss_1˚<@��/O5       ��]�	T������A�*'

learning_rate_1���:

loss_1.�G@�<�M5       ��]�	.m�����A�*'

learning_rate_1���:

loss_1ZoM@�"�75       ��]�	������A�*'

learning_rate_1���:

loss_1$�E@�S��5       ��]�	�-�����A�*'

learning_rate_1���:

loss_1gG@'�o5       ��]�	�������A�*'

learning_rate_1���:

loss_1��:@�a�5       ��]�	������A�*'

learning_rate_1���:

loss_1^�N@Տ5       ��]�	��&����A�*'

learning_rate_1���:

loss_1��>@/ӝ&5       ��]�	�R>����A�*'

learning_rate_1���:

loss_1*@XP��5       ��]�	7�T����A�*'

learning_rate_1���:

loss_12�K@ȄK�5       ��]�	��l����A�*'

learning_rate_1���:

loss_1�37@Kz2 5       ��]�	�R�����A�*'

learning_rate_1���:

loss_1w6L@�GM5       ��]�	q�����A�*'

learning_rate_1���:

loss_1/=@�: �5       ��]�	�0�����A�*'

learning_rate_1���:

loss_1l3A@��X$5       ��]�	�������A�*'

learning_rate_1���:

loss_1��;@;n �5       ��]�	R������A�*'

learning_rate_1���:

loss_1�K@;��5       ��]�	�X�����A�*'

learning_rate_1���:

loss_1�G@�z 5       ��]�	�y����A�*'

learning_rate_1���:

loss_1�L@���Z5       ��]�	y�����A�*'

learning_rate_1���:

loss_1��<@�ծB5       ��]�	�5����A�*'

learning_rate_1���:

loss_1gC@G�[!5       ��]�	aK����A�*'

learning_rate_1���:

loss_1�iC@��25       ��]�	ņa����A�*'

learning_rate_1���:

loss_1��G@���F5       ��]�	o�w����A�*'

learning_rate_1���:

loss_1ܚ>@��7�5       ��]�	8�����A�*'

learning_rate_1���:

loss_14�K@@��5       ��]�	☤����A�*'

learning_rate_1���:

loss_1	xI@C���5       ��]�	�������A�*'

learning_rate_1���:

loss_1)SQ@�r5       ��]�	�&�����A�*'

learning_rate_1���:

loss_1)�<@B[a�5       ��]�	�H�����A�*'

learning_rate_1���:

loss_1v@@���5       ��]�	A������A�*'

learning_rate_1���:

loss_1?@�ˊ�5       ��]�	u�����A�*'

learning_rate_1���:

loss_1b�K@jd��5       ��]�	�)����A�*'

learning_rate_1���:

loss_1Pn;@��/5       ��]�	�)?����A�*'

learning_rate_1���:

loss_1�u>@�H�5       ��]�	��V����A�*'

learning_rate_1���:

loss_1Y$:@�˭x5       ��]�	�
m����A�*'

learning_rate_1���:

loss_1c�:@F�5       ��]�	z=�����A�*'

learning_rate_1���:

loss_1x�5@ m>Y5       ��]�	�w�����A�*'

learning_rate_1���:

loss_1x�@@x�=�5       ��]�	�������A�*'

learning_rate_1���:

loss_1�3=@�ri5       ��]�	$������A�*'

learning_rate_1���:

loss_1�I@I�~�5       ��]�	�������A�*'

learning_rate_1���:

loss_1�AJ@��5       ��]�	�5�����A�*'

learning_rate_1���:

loss_1v:E@"M�5       ��]�	�����A�*'

learning_rate_1���:

loss_1A�/@�X�x5       ��]�	׹#����A�*'

learning_rate_1���:

loss_1fH@Nw��5       ��]�	��:����A�*'

learning_rate_1���:

loss_1�R;@_�~�5       ��]�	�P����A�*'

learning_rate_1���:

loss_1m�3@���f5       ��]�	�^g����A�*'

learning_rate_1���:

loss_1G�G@wRV5       ��]�	Z�}����A�*'

learning_rate_1���:

loss_1��L@;�C5       ��]�	<0�����A�*'

learning_rate_1���:

loss_1%64@C6�	5       ��]�	������A�*'

learning_rate_1���:

loss_1-1@��+�5       ��]�	�U�����A�*'

learning_rate_1���:

loss_1r�E@��X�5       ��]�	������A�*'

learning_rate_1���:

loss_1�?@b�ߜ5       ��]�	K
�����A�*'

learning_rate_1���:

loss_1�@@TX��5       ��]�	�;����A�*'

learning_rate_1���:

loss_1);E@�خ�5       ��]�	�|����A�*'

learning_rate_1���:

loss_1��;@�9q�5       ��]�	�4����A�*'

learning_rate_1���:

loss_1�KU@n�&m5       ��]�	w�J����A�*'

learning_rate_1���:

loss_1@bG@IY��5       ��]�	�`����A�*'

learning_rate_1���:

loss_1!{E@���5       ��]�	!Ox����A�*'

learning_rate_1���:

loss_1�iJ@��5       ��]�	�������A�*'

learning_rate_1���:

loss_1�xJ@C._�5       ��]�	8?�����A�*'

learning_rate_1���:

loss_1��K@oxy�5       ��]�	�������A�*'

learning_rate_1���:

loss_1I\C@n��5       ��]�	�������A�*'

learning_rate_1���:

loss_1�=@�Gw]5       ��]�	�������A�*'

learning_rate_1���:

loss_1�J@��C�5       ��]�	o�����A�*'

learning_rate_1���:

loss_1�v<@��
5       ��]�	�;����A�*'

learning_rate_1���:

loss_1CYI@���5       ��]�	��+����A�*'

learning_rate_1���:

loss_1� C@d���5       ��]�	d�A����A�*'

learning_rate_1���:

loss_1L;@д/�5       ��]�	�X����A�*'

learning_rate_1���:

loss_1XA@��r�5       ��]�	=;q����A�*'

learning_rate_1���:

loss_1�;G@�
nG5       ��]�	�<�����A�*'

learning_rate_1���:

loss_1��F@%=�(5       ��]�	u������A�*'

learning_rate_1���:

loss_1(�B@���U5       ��]�	�˳����A�*'

learning_rate_1���:

loss_1yF@8�5       ��]�	������A�*'

learning_rate_1���:

loss_1�kF@��5       ��]�	�������A�*'

learning_rate_1���:

loss_1	]>@��C5       ��]�	������A�*'

learning_rate_1���:

loss_1x�M@��U5       ��]�	������A�*'

learning_rate_1���:

loss_1~�:@A�B5       ��]�	u%$����A�*'

learning_rate_1���:

loss_1S�D@kؚ<5       ��]�	.o:����A�*'

learning_rate_1���:

loss_1�J@�;t�5       ��]�	�P����A�*'

learning_rate_1���:

loss_1�L?@к�$5       ��]�	5�f����A�*'

learning_rate_1���:

loss_1��C@\�5       ��]�	�P}����A�*'

learning_rate_1���:

loss_1U
F@�䨛5       ��]�	�������A�*'

learning_rate_1���:

loss_1�9@R{�5       ��]�	�$�����A�*'

learning_rate_1���:

loss_1�_L@f_0�5       ��]�	������A�*'

learning_rate_1���:

loss_1��H@��r�5       ��]�	�O�����A�*'

learning_rate_1���:

loss_1��:@QK��5       ��]�	V|�����A�*'

learning_rate_1���:

loss_1՟B@b�75       ��]�	 �����A�*'

learning_rate_1���:

loss_1�G@]G�=5       ��]�	M�����A�*'

learning_rate_1���:

loss_1wD@�l{�5       ��]�	-�/����A�*'

learning_rate_1���:

loss_1~�A@[�||5       ��]�	�?F����A�*'

learning_rate_1���:

loss_14B@]�G5       ��]�	Z�\����A�*'

learning_rate_1���:

loss_18�K@��5       ��]�	�s����A�*'

learning_rate_1���:

loss_1BF@�M�h5       ��]�	Q�����A�*'

learning_rate_1���:

loss_1��L@!{�%5       ��]�	Rv�����A�*'

learning_rate_1���:

loss_1�}D@hSK5       ��]�	nǶ����A�*'

learning_rate_1���:

loss_1��?@Do|`5       ��]�	������A�*'

learning_rate_1���:

loss_1�O@���m5       ��]�	������A�*'

learning_rate_1���:

loss_1��A@�U�5       ��]�	�"�����A�*'

learning_rate_1���:

loss_1�>@�[�5       ��]�	]i����A�*'

learning_rate_1���:

loss_1`�:@eu.5       ��]�	/�'����A�*'

learning_rate_1���:

loss_1V1@�/��5       ��]�	�=����A�*'

learning_rate_1���:

loss_1_�K@��}5       ��]�	�[T����A�*'

learning_rate_1���:

loss_1��F@��a5       ��]�	e�j����A�*'

learning_rate_1���:

loss_1-?@6��a5       ��]�	�ɀ����A�*'

learning_rate_1���:

loss_1>=G@��<�5       ��]�	������A�*'

learning_rate_1���:

loss_1(�H@P�:5       ��]�	x�����A�*'

learning_rate_1���:

loss_1�tH@$���5       ��]�	�������A�*'

learning_rate_1���:

loss_1�I@E��5       ��]�	�������A�*'

learning_rate_1���:

loss_1r�?@��B�5       ��]�	�������A�*'

learning_rate_1���:

loss_1�;@w��n5       ��]�	�L����A�*'

learning_rate_1���:

loss_182@����5       ��]�	 #����A�*'

learning_rate_1���:

loss_1ވ=@��F5       ��]�	��/����A�*'

learning_rate_1���:

loss_1�<V@�+��5       ��]�	�F����A�*'

learning_rate_1���:

loss_1B�O@J��5       ��]�	�z\����A�*'

learning_rate_1���:

loss_1�,E@� W5       ��]�	��r����A�*'

learning_rate_1���:

loss_1�D@�쉣5       ��]�	������A�*'

learning_rate_1���:

loss_1�5=@ؘ�S5       ��]�	(q�����A�*'

learning_rate_1���:

loss_1ցL@ƛX�5       ��]�	�3�����A�*'

learning_rate_1���:

loss_160C@y,�5       ��]�	�������A�*'

learning_rate_1���:

loss_1|�F@]�=�5       ��]�	�������A�*'

learning_rate_1���:

loss_1��H@|�a5       ��]�	�0�����A�*'

learning_rate_1���:

loss_1�?C@��5�5       ��]�	�a����A�*'

learning_rate_1���:

loss_1��W@А��5       ��]�	�&����A�*'

learning_rate_1���:

loss_1�K@K6b�5       ��]�	k?����A�*'

learning_rate_1���:

loss_1�>2@碂5       ��]�	N<V����A�*'

learning_rate_1���:

loss_1��3@Y�5       ��]�	�|l����A�*'

learning_rate_1���:

loss_1�A@~�`�5       ��]�	������A�*'

learning_rate_1���:

loss_1C�I@��*;5       ��]�	A%�����A�*'

learning_rate_1���:

loss_1?@i�I�5       ��]�	�������A�*'

learning_rate_1���:

loss_1(;I@>!G�5       ��]�	�������A�*'

learning_rate_1���:

loss_1-)H@�tu�5       ��]�	� �����A�*'

learning_rate_1���:

loss_1�sC@'"+5       ��]�	4%�����A�*'

learning_rate_1���:

loss_1ֶD@���5       ��]�	L\	����A�*'

learning_rate_1���:

loss_1GNN@�L�}5       ��]�	������A�*'

learning_rate_1���:

loss_1�N@��T5       ��]�	1	6����A�*'

learning_rate_1���:

loss_1�3A@�o8�5       ��]�	$~L����A�*'

learning_rate_1���:

loss_1I�E@*@]5       ��]�	��b����A�*'

learning_rate_1���:

loss_1�4;@1=�5       ��]�	D�x����A�*'

learning_rate_1���:

loss_1ANQ@���5       ��]�	x�����A�*'

learning_rate_1���:

loss_1�Q:@�V�5       ��]�	�D�����A�*'

learning_rate_1���:

loss_1�Q?@��c5       ��]�	>h�����A�*'

learning_rate_1���:

loss_1��M@;��5       ��]�	�������A�*'

learning_rate_1���:

loss_1B"@@ �@�5       ��]�	�������A�*'

learning_rate_1���:

loss_1V�>@�x;:5       ��]�	�� ���A�*'

learning_rate_1���:

loss_1��R@
 �5       ��]�	�& ���A�*'

learning_rate_1���:

loss_1��M@Є�g5       ��]�	��. ���A�*'

learning_rate_1���:

loss_1�h7@g�]�5       ��]�	(#E ���A�*'

learning_rate_1���:

loss_1FW@�jG5       ��]�	�] ���A�*'

learning_rate_1���:

loss_1�D@u�t5       ��]�	S�s ���A�*'

learning_rate_1���:

loss_1�>@���5       ��]�	�҉ ���A�*'

learning_rate_1���:

loss_1QtB@U-�5       ��]�	�[� ���A�*'

learning_rate_1���:

loss_1�;@	H�}5       ��]�	�Ƿ ���A�*'

learning_rate_1���:

loss_1��F@�I%5       ��]�	�\� ���A�*'

learning_rate_1���:

loss_1-lN@���5       ��]�	'�� ���A�*'

learning_rate_1���:

loss_1��P@���5       ��]�	q0� ���A�*'

learning_rate_1���:

loss_1ҴL@��45       ��]�	����A�*'

learning_rate_1���:

loss_1�FV@�lZ5       ��]�	�E%���A�*'

learning_rate_1���:

loss_1 J@E��5       ��]�	��;���A�*'

learning_rate_1���:

loss_1��M@�\5       ��]�	��Q���A�*'

learning_rate_1���:

loss_1�KA@�o��5       ��]�	z:i���A�*'

learning_rate_1���:

loss_1S�E@)#��5       ��]�	X���A�*'

learning_rate_1���:

loss_1K@��S�5       ��]�	�t����A�*'

learning_rate_1���:

loss_1U�G@|�}?5       ��]�	/�����A�*'

learning_rate_1���:

loss_1{%L@e�k�5       ��]�	�����A�*'

learning_rate_1���:

loss_1�6@>6�5       ��]�	�����A�*'

learning_rate_1���:

loss_1�EB@T�or5       ��]�	�����A�*'

learning_rate_1���:

loss_1�`F@�2�a5       ��]�	�����A�*'

learning_rate_1���:

loss_1#�G@�0��5       ��]�	�����A�*'

learning_rate_1���:

loss_1jE9@���5       ��]�	n�3���A�*'

learning_rate_1���:

loss_1�~<@ʠ:�5       ��]�	XJ���A�*'

learning_rate_1���:

loss_19WA@4ZT5       ��]�	�X`���A�*'

learning_rate_1���:

loss_1.�E@E_A�5       ��]�	ۋv���A�*'

learning_rate_1���:

loss_1�/G@���5       ��]�	ԫ����A�*'

learning_rate_1���:

loss_1K�K@ɑ��5       ��]�	�����A�*'

learning_rate_1���:

loss_1��E@�W��5       ��]�	�����A�*'

learning_rate_1���:

loss_1�8@��/�5       ��]�	�_����A�*'

learning_rate_1���:

loss_1��K@�Sy}5       ��]�	������A�*'

learning_rate_1���:

loss_1%ME@��^�5       ��]�	!����A�*'

learning_rate_1���:

loss_1��@@�lRQ5       ��]�	�����A�*'

learning_rate_1���:

loss_1]�R@�0�5       ��]�	b�)���A�*'

learning_rate_1���:

loss_1�<D@�gg5       ��]�	?YA���A�*'

learning_rate_1���:

loss_1�|7@N��55       ��]�	}�W���A�*'

learning_rate_1���:

loss_1�EM@p��5       ��]�	��m���A�*'

learning_rate_1���:

loss_1�)A@�nN�5       ��]�	�����A�*'

learning_rate_1���:

loss_1]>@@c��5       ��]�	�o����A�*'

learning_rate_1���:

loss_1�[T@�?��5       ��]�	°���A�*'

learning_rate_1���:

loss_1#�F@I߱5       ��]�	�-����A�*'

learning_rate_1���:

loss_1'�D@M���5       ��]�	�����A�*'

learning_rate_1���:

loss_1ClK@W,7�5       ��]�	Z6����A�*'

learning_rate_1���:

loss_1��I@5���5       ��]�	Wq	���A�*'

learning_rate_1���:

loss_1�wK@��5       ��]�	[9���A�*'

learning_rate_1���:

loss_1��B@��S5       ��]�	35���A�*'

learning_rate_1���:

loss_1�9@�I�+5       ��]�	�ZK���A�*'

learning_rate_1���:

loss_1�DJ@+�2:5       ��]�	שa���A�*'

learning_rate_1���:

loss_1X�K@u�:V5       ��]�	Ƌw���A�*'

learning_rate_1���:

loss_1j�@@t�O�5       ��]�	Hԍ���A�*'

learning_rate_1���:

loss_1y�@@��c5       ��]�	ۣ���A�*'

learning_rate_1���:

loss_1�9M@>G�5       ��]�	�H����A�*'

learning_rate_1���:

loss_1�C@1�`5       ��]�	l�����A�*'

learning_rate_1���:

loss_1�I@}��5       ��]�	������A�*'

learning_rate_1���:

loss_1��D@H}T�5       ��]�	$����A�*'

learning_rate_1���:

loss_1��2@�=kd5       ��]�	x@���A�*'

learning_rate_1���:

loss_1�?@�\5       ��]�	�O*���A�*'

learning_rate_1���:

loss_1��B@��&>5       ��]�	��@���A�*'

learning_rate_1���:

loss_1��@@���;5       ��]�	*W���A�*'

learning_rate_1���:

loss_1gc>@���.5       ��]�	��m���A�*'

learning_rate_1���:

loss_1/O@@���5       ��]�	u����A�*'

learning_rate_1���:

loss_1�rF@�t�5       ��]�	ł����A�*'

learning_rate_1���:

loss_1}E@q5       ��]�	Q^����A�*'

learning_rate_1���:

loss_1P�T@�N�5       ��]�	������A�*'

learning_rate_1���:

loss_1c�:@A��a5       ��]�	�)����A�*'

learning_rate_1���:

loss_1�4B@3�j�5       ��]�	a����A�*'

learning_rate_1���:

loss_1�M@�A��5       ��]�	����A�*'

learning_rate_1���:

loss_1P�<@'�HF5       ��]�	�_$���A�*'

learning_rate_1���:

loss_1�[T@ "h05       ��]�	]�:���A�*'

learning_rate_1���:

loss_1��?@x	�45       ��]�	s�P���A�*'

learning_rate_1���:

loss_1��@@����5       ��]�	�g���A�*'

learning_rate_1���:

loss_1�$D@�#>�5       ��]�	=/}���A�*'

learning_rate_1���:

loss_1��I@��5       ��]�	�Ɠ���A�*'

learning_rate_1���:

loss_1��'@;>�5       ��]�	����A�*'

learning_rate_1���:

loss_1�%:@�?�5       ��]�	�F����A�*'

learning_rate_1���:

loss_1C@Y %y5       ��]�	�k����A�*'

learning_rate_1���:

loss_1ʪC@SQ�5       ��]�	�����A�*'

learning_rate_1���:

loss_1�<@��D!5       ��]�	�F���A�*'

learning_rate_1���:

loss_1)M@�u�5       ��]�	�,���A�*'

learning_rate_1���:

loss_1 B@A�*5       ��]�	�L1���A�*'

learning_rate_1���:

loss_1�L@@8�}S5       ��]�	W�G���A�*'

learning_rate_1���:

loss_1t�E@��5       ��]�	g^���A�*'

learning_rate_1���:

loss_1v�H@��'�5       ��]�	Wet���A�*'

learning_rate_1���:

loss_150N@���5       ��]�	�����A�*'

learning_rate_1���:

loss_1_M@�2ah5       ��]�	q����A�*'

learning_rate_1���:

loss_1�x@@���5       ��]�	�n����A�*'

learning_rate_1���:

loss_1��K@ɻ:5       ��]�	0-����A�*'

learning_rate_1���:

loss_1��E@M��5       ��]�	�a����A�*'

learning_rate_1���:

loss_1*�K@@��5       ��]�	������A�*'

learning_rate_1���:

loss_1�@=@TS/5       ��]�	�����A�*'

learning_rate_1���:

loss_1��@@���5       ��]�	��%���A�*'

learning_rate_1���:

loss_1��V@��\�5       ��]�	��;���A�*'

learning_rate_1���:

loss_1�K@�>�i5       ��]�	�6R���A�*'

learning_rate_1���:

loss_1�T@v`�a5       ��]�	p�h���A�*'

learning_rate_1���:

loss_1�F@o�jC5       ��]�	kۀ���A�*'

learning_rate_1���:

loss_1"^C@{UF5       ��]�	<O����A�*'

learning_rate_1���:

loss_1�dN@�=g�5       ��]�	������A�*'

learning_rate_1���:

loss_1��R@ ��5       ��]�	������A�*'

learning_rate_1���:

loss_1�D@�NԢ5       ��]�	U�����A�*'

learning_rate_1���:

loss_15O@�A�?5       ��]�	<�����A�*'

learning_rate_1���:

loss_1�MG@=O��5       ��]�	{	���A�*'

learning_rate_1���:

loss_1��>@�9��5       ��]�	�	���A�*'

learning_rate_1���:

loss_1��:@��5       ��]�	��4	���A�*'

learning_rate_1���:

loss_1,mD@'&�5       ��]�	�K	���A�*'

learning_rate_1���:

loss_1C�B@���U5       ��]�	��b	���A�*'

learning_rate_1���:

loss_1�X=@7�&5       ��]�	�.x	���A�*'

learning_rate_1���:

loss_1+A@���N5       ��]�	�َ	���A�*'

learning_rate_1���:

loss_1�?@ߌ�:5       ��]�	?�	���A�*'

learning_rate_1���:

loss_1F�5@F=`�5       ��]�	[h�	���A�*'

learning_rate_1���:

loss_1~�J@��%�5       ��]�	T�	���A�*'

learning_rate_1���:

loss_1��9@�DC�5       ��]�	x��	���A�*'

learning_rate_1���:

loss_1ӣR@�*��5       ��]�	� 
���A�*'

learning_rate_1���:

loss_1�O@�5Ё5       ��]�	N
���A�*'

learning_rate_1���:

loss_1*?@��z5       ��]�	5�,
���A�*'

learning_rate_1���:

loss_1�gL@P&M5       ��]�	<&D
���A�*'

learning_rate_1���:

loss_1�6@ٷ�C5       ��]�	�RZ
���A�*'

learning_rate_1���:

loss_1H�8@h��5       ��]�	�q
���A�*'

learning_rate_1���:

loss_1rdK@&�A�5       ��]�	�e�
���A�*'

learning_rate_1���:

loss_1n�H@�H�5       ��]�	]Z�
���A�*'

learning_rate_1���:

loss_1�VN@�5       ��]�	G?�
���A�*'

learning_rate_1���:

loss_1mI@�V�L5       ��]�	t�
���A�*'

learning_rate_1���:

loss_1�(:@�/�:5       ��]�	DU�
���A�*'

learning_rate_1���:

loss_1/�G@�avY5       ��]�	f�
���A�*'

learning_rate_1���:

loss_1��H@%!�5       ��]�	t����A�*'

learning_rate_1���:

loss_1��K@��v 5       ��]�	|@&���A�*'

learning_rate_1���:

loss_1��G@����5       ��]�	Xp<���A�*'

learning_rate_1���:

loss_1 �@@)��.5       ��]�	��S���A�*'

learning_rate_1���:

loss_1��F@�T.�5       ��]�	/�i���A�*'

learning_rate_1���:

loss_1S�K@���.5       ��]�	�0����A�*'

learning_rate_1���:

loss_1z7@�`|�5       ��]�	V�����A�*'

learning_rate_1���:

loss_1��N@����5       ��]�	�����A�*'

learning_rate_1���:

loss_1KC@"�v5       ��]�	�����A�*'

learning_rate_1���:

loss_1�zN@�xK�5       ��]�	�����A�*'

learning_rate_1���:

loss_1��?@��Q5       ��]�	 l����A�*'

learning_rate_1���:

loss_1E[M@]x4�5       ��]�	�����A�*'

learning_rate_1���:

loss_1̞E@/�5       ��]�	�_���A�*'

learning_rate_1���:

loss_1FC@��5       ��]�	�T3���A�*'

learning_rate_1���:

loss_1�:@8I��5       ��]�	��I���A�*'

learning_rate_1���:

loss_1�F@��25       ��]�	�`���A�*'

learning_rate_1���:

loss_1#)A@;���5       ��]�	��v���A�*'

learning_rate_1���:

loss_1V<@OMb�5       ��]�	M����A�*'

learning_rate_1���:

loss_1��B@�;P+5       ��]�	������A�*'

learning_rate_1���:

loss_1��J@!�P5       ��]�	�B����A�*'

learning_rate_1���:

loss_1ugH@48ŝ5       ��]�	'����A�*'

learning_rate_1 t�:

loss_1(Q@�|�
5       ��]�	>����A�*'

learning_rate_1 t�:

loss_1h?;@��'5       ��]�	G=���A�*'

learning_rate_1 t�:

loss_1C�C@ʡ��5       ��]�	�h���A�*'

learning_rate_1 t�:

loss_1��E@;���5       ��]�	��2���A�*'

learning_rate_1 t�:

loss_1�	5@�)��5       ��]�	/�H���A�*'

learning_rate_1 t�:

loss_16>@��S5       ��]�	"�`���A�*'

learning_rate_1 t�:

loss_1K�A@�2[�5       ��]�	��v���A�*'

learning_rate_1 t�:

loss_1�GG@��;�5       ��]�	xN����A�*'

learning_rate_1 t�:

loss_1B/I@�#�5       ��]�	�)����A�*'

learning_rate_1 t�:

loss_1%X@:U`@5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1��=@��1\5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1��B@4W<5       ��]�	�M����A�*'

learning_rate_1 t�:

loss_1>JD@����5       ��]�	������A�*'

learning_rate_1 t�:

loss_1�=@L_M�5       ��]�	?����A�*'

learning_rate_1 t�:

loss_1�O@�ї�5       ��]�	�9'���A�*'

learning_rate_1 t�:

loss_1v�C@�T��5       ��]�	)v=���A�*'

learning_rate_1 t�:

loss_18H@S�z!5       ��]�	W�S���A�*'

learning_rate_1 t�:

loss_1�M@�ƌq5       ��]�	�yj���A�*'

learning_rate_1 t�:

loss_1T�E@ŕph5       ��]�	+m����A�*'

learning_rate_1 t�:

loss_1oD@¹�f5       ��]�	������A�*'

learning_rate_1 t�:

loss_1��<@���~5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1UH@;W�5       ��]�	c�����A�*'

learning_rate_1 t�:

loss_1wQ@s�5�5       ��]�	ʌ����A�*'

learning_rate_1 t�:

loss_1gD@����5       ��]�	������A�*'

learning_rate_1 t�:

loss_1�=@DK�j5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1+O@���5       ��]�	`����A�*'

learning_rate_1 t�:

loss_1�7G@���5       ��]�	N�3���A�*'

learning_rate_1 t�:

loss_1|�=@$cb�5       ��]�	��J���A�*'

learning_rate_1 t�:

loss_1�;B@+�5       ��]�	b���A�*'

learning_rate_1 t�:

loss_1'�2@��s5       ��]�	�x���A�*'

learning_rate_1 t�:

loss_1��G@jvn5       ��]�	�n����A�*'

learning_rate_1 t�:

loss_1�46@��K5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1�V@@�ʚ�5       ��]�	�%����A�*'

learning_rate_1 t�:

loss_1Ǉ4@Z�oQ5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1�L<@zdW�5       ��]�	������A�*'

learning_rate_1 t�:

loss_1U�@@�,�5       ��]�	R*���A�*'

learning_rate_1 t�:

loss_1L�9@��Ka5       ��]�	����A�*'

learning_rate_1 t�:

loss_1=�?@*�;5       ��]�	�4-���A�*'

learning_rate_1 t�:

loss_1�>@��5       ��]�	7�C���A�*'

learning_rate_1 t�:

loss_1�A@�(�5       ��]�	lAZ���A�*'

learning_rate_1 t�:

loss_1��=@��|5       ��]�	��p���A�*'

learning_rate_1 t�:

loss_1h�H@�%5       ��]�	ӆ���A�*'

learning_rate_1 t�:

loss_1-�B@�6<�5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1'hD@��*�5       ��]�	/d����A�*'

learning_rate_1 t�:

loss_1*�C@E6�5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1ـ;@�P�45       ��]�	r�����A�*'

learning_rate_1 t�:

loss_1,<B@D�5       ��]�	ʩ����A�*'

learning_rate_1 t�:

loss_1��7@b�+�5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1�O@[!�5       ��]�	%���A�*'

learning_rate_1 t�:

loss_1��J@����5       ��]�	U�;���A�*'

learning_rate_1 t�:

loss_1v{D@�mk�5       ��]�	�Q���A�*'

learning_rate_1 t�:

loss_1��F@�/�5       ��]�	#h���A�*'

learning_rate_1 t�:

loss_1��?@+~�d5       ��]�	�`~���A�*'

learning_rate_1 t�:

loss_1��A@�Y75       ��]�	F����A�*'

learning_rate_1 t�:

loss_1~5@��ǟ5       ��]�	Q����A�*'

learning_rate_1 t�:

loss_1�D@�qJ�5       ��]�	������A�*'

learning_rate_1 t�:

loss_14�M@���5       ��]�	�7����A�*'

learning_rate_1 t�:

loss_1�]3@�گ5       ��]�	�I����A�*'

learning_rate_1 t�:

loss_1N@YAW�5       ��]�	g����A�*'

learning_rate_1 t�:

loss_1��2@��a�5       ��]�	~���A�*'

learning_rate_1 t�:

loss_1�6H@��C�5       ��]�	$�4���A�*'

learning_rate_1 t�:

loss_1{�4@���5       ��]�	�1K���A�*'

learning_rate_1 t�:

loss_1B�<@�Ki�5       ��]�	�	b���A�*'

learning_rate_1 t�:

loss_1f�5@-��5       ��]�	?y���A�*'

learning_rate_1 t�:

loss_1�:@xq5       ��]�	]����A�*'

learning_rate_1 t�:

loss_1��F@P�5       ��]�	������A�*'

learning_rate_1 t�:

loss_1}cE@�b�F5       ��]�	�ؼ���A�*'

learning_rate_1 t�:

loss_1��C@T���5       ��]�	�����A�*'

learning_rate_1 t�:

loss_1t&G@��_�5       ��]�	i�����A�*'

learning_rate_1 t�:

loss_1sT6@�<*5       ��]�	�h���A�*'

learning_rate_1 t�:

loss_1oJ8@TQ�5       ��]�	l����A�*'

learning_rate_1 t�:

loss_1�?@g��5       ��]�	��-���A�*'

learning_rate_1 t�:

loss_1'�=@]�5       ��]�	k�E���A�*'

learning_rate_1 t�:

loss_1�?@#v��5       ��]�	��[���A�*'

learning_rate_1 t�:

loss_1��D@kڠ:5       ��]�	�r���A�*'

learning_rate_1 t�:

loss_1UN@+6Y�5       ��]�	�q����A�*'

learning_rate_1 t�:

loss_1��L@�f?�5       ��]�	Hy����A�*'

learning_rate_1 t�:

loss_1��<@���5       ��]�	N����A�*'

learning_rate_1 t�:

loss_1��>@���5       ��]�	�H����A�*'

learning_rate_1 t�:

loss_1�B@nG3�5       ��]�	 |����A�*'

learning_rate_1 t�:

loss_1�xH@\�v5       ��]�	u����A�*'

learning_rate_1 t�:

loss_1�YI@��?85       ��]�	�����A�*'

learning_rate_1 t�:

loss_1��E@�C��5       ��]�	F�%���A�*'

learning_rate_1 t�:

loss_1I�I@���x5       ��]�	H<���A�*'

learning_rate_1 t�:

loss_1��@@'Ӝ5       ��]�	R�R���A�*'

learning_rate_1 t�:

loss_1��@@+�5       ��]�	� i���A�*'

learning_rate_1 t�:

loss_1��K@7K��5       ��]�	M���A�*'

learning_rate_1 t�:

loss_1�mM@���85       ��]�	����A�*'

learning_rate_1 t�:

loss_1�L@B
w5       ��]�	������A�*'

learning_rate_1 t�:

loss_1�-@��j5       ��]�	ė����A�*'

learning_rate_1 t�:

loss_1��;@�yA5       ��]�	`�����A�*'

learning_rate_1 t�:

loss_1�WB@�b�5       ��]�	f�����A�*'

learning_rate_1 t�:

loss_1��8@�v��5       ��]�	r����A�*'

learning_rate_1 t�:

loss_1�'O@�2��5       ��]�	
6���A�*'

learning_rate_1 t�:

loss_1�G@���5       ��]�	"m4���A�*'

learning_rate_1 t�:

loss_1xM@��5       ��]�	]�J���A� *'

learning_rate_1 t�:

loss_1&\E@~��5       ��]�	�c���A� *'

learning_rate_1 t�:

loss_1�J7@$5�5       ��]�	�[y���A� *'

learning_rate_1 t�:

loss_1�15@��0&5       ��]�	�Ï���A� *'

learning_rate_1 t�:

loss_1�>@I�Y5       ��]�	,�����A� *'

learning_rate_1 t�:

loss_1!M@����5       ��]�	G����A� *'

learning_rate_1 t�:

loss_1p�C@-�5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1xc=@m�N%5       ��]�	�?����A� *'

learning_rate_1 t�:

loss_1�C6@�*�5       ��]�	Ț ���A� *'

learning_rate_1 t�:

loss_1ֵO@�{k5       ��]�	[]���A� *'

learning_rate_1 t�:

loss_1=�C@�;��5       ��]�	P�-���A� *'

learning_rate_1 t�:

loss_1�8@��@5       ��]�	�+D���A� *'

learning_rate_1 t�:

loss_1�K@8�A�5       ��]�	8\Z���A� *'

learning_rate_1 t�:

loss_1�2;@�5 $5       ��]�	��p���A� *'

learning_rate_1 t�:

loss_1K(>@���5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1֡9@\|��5       ��]�	W����A� *'

learning_rate_1 t�:

loss_1�8=@�e�J5       ��]�	�?����A� *'

learning_rate_1 t�:

loss_1��G@�2 !5       ��]�	ˑ����A� *'

learning_rate_1 t�:

loss_1vc:@؆��5       ��]�	������A� *'

learning_rate_1 t�:

loss_1�=@��X�5       ��]�	������A� *'

learning_rate_1 t�:

loss_1hR6@Ŧ�5       ��]�	]���A� *'

learning_rate_1 t�:

loss_1gQ@ԥ�B5       ��]�	h:#���A� *'

learning_rate_1 t�:

loss_1��T@lr�5       ��]�	J�9���A� *'

learning_rate_1 t�:

loss_1 \7@�ym25       ��]�	C�O���A� *'

learning_rate_1 t�:

loss_1u{G@��55       ��]�	��e���A� *'

learning_rate_1 t�:

loss_1��:@��_5       ��]�	{|���A� *'

learning_rate_1 t�:

loss_1k(I@�N}�5       ��]�	|����A� *'

learning_rate_1 t�:

loss_1�G@�O+5       ��]�	6�����A� *'

learning_rate_1 t�:

loss_1j�@@�-��5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1I�8@���5       ��]�	�=����A� *'

learning_rate_1 t�:

loss_1̓C@���D5       ��]�	�g����A� *'

learning_rate_1 t�:

loss_1I<@���5       ��]�	ǽ���A� *'

learning_rate_1 t�:

loss_1�a9@���t5       ��]�	E����A� *'

learning_rate_1 t�:

loss_1��;@�K�O5       ��]�	��.���A� *'

learning_rate_1 t�:

loss_12�F@P��%5       ��]�	�AE���A� *'

learning_rate_1 t�:

loss_1(D@̳E�5       ��]�	��[���A� *'

learning_rate_1 t�:

loss_1��>@\8P�5       ��]�	f�q���A� *'

learning_rate_1 t�:

loss_1 �N@V��5       ��]�	C`����A� *'

learning_rate_1 t�:

loss_1I�E@mao5       ��]�	I����A� *'

learning_rate_1 t�:

loss_1�i@@܌R5       ��]�	�]����A� *'

learning_rate_1 t�:

loss_1��;@�ڃ5       ��]�	�b����A� *'

learning_rate_1 t�:

loss_1��G@�7��5       ��]�	������A� *'

learning_rate_1 t�:

loss_1�7G@6
�5       ��]�	������A� *'

learning_rate_1 t�:

loss_1�,@�p�5       ��]�	�y���A� *'

learning_rate_1 t�:

loss_1�S5@�G5       ��]�	f�#���A� *'

learning_rate_1 t�:

loss_1��8@a���5       ��]�	��9���A� *'

learning_rate_1 t�:

loss_1\�?@�р5       ��]�	�mP���A� *'

learning_rate_1 t�:

loss_1#�>@�Z`5       ��]�	<Xh���A� *'

learning_rate_1 t�:

loss_1v}J@��5       ��]�	p�~���A� *'

learning_rate_1 t�:

loss_1Q�L@#O.S5       ��]�	D����A� *'

learning_rate_1 t�:

loss_1�I@�؜n5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1�>@S���5       ��]�	������A� *'

learning_rate_1 t�:

loss_1a�Q@���5       ��]�	77����A� *'

learning_rate_1 t�:

loss_1�H@iI8$5       ��]�	g�����A� *'

learning_rate_1 t�:

loss_1�;@THI5       ��]�	b����A� *'

learning_rate_1 t�:

loss_1' 8@�ڹ�5       ��]�	P!���A� *'

learning_rate_1 t�:

loss_1�sM@�@�*5       ��]�	�U2���A� *'

learning_rate_1 t�:

loss_1��6@I�T�5       ��]�	�I���A� *'

learning_rate_1 t�:

loss_1w�?@�`�5       ��]�	�=_���A� *'

learning_rate_1 t�:

loss_1:>@��K5       ��]�	�u���A� *'

learning_rate_1 t�:

loss_1�@@wI�L5       ��]�	�i����A� *'

learning_rate_1 t�:

loss_1�VB@��f5       ��]�	+Ģ���A� *'

learning_rate_1 t�:

loss_1�XL@����5       ��]�	�	����A� *'

learning_rate_1 t�:

loss_1z9I@K�v5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1�-@'��5       ��]�	�M����A� *'

learning_rate_1 t�:

loss_1�xB@j7��5       ��]�	������A� *'

learning_rate_1 t�:

loss_1�8@Yd	�5       ��]�	?D���A� *'

learning_rate_1 t�:

loss_1a�C@����5       ��]�	Sl+���A� *'

learning_rate_1 t�:

loss_1LkJ@8�>�5       ��]�	��A���A� *'

learning_rate_1 t�:

loss_1p:@���65       ��]�	�UY���A� *'

learning_rate_1 t�:

loss_1�5@T�y�5       ��]�	I�o���A� *'

learning_rate_1 t�:

loss_1�8@J��5       ��]�	�Յ���A� *'

learning_rate_1 t�:

loss_1��K@,�O5       ��]�	$�����A� *'

learning_rate_1 t�:

loss_13�B@]VGS5       ��]�	ͭ����A� *'

learning_rate_1 t�:

loss_1;=@��5       ��]�	������A� *'

learning_rate_1 t�:

loss_14B@ŋ�5       ��]�	�C����A� *'

learning_rate_1 t�:

loss_1�9@�e��5       ��]�	d����A� *'

learning_rate_1 t�:

loss_1ʢA@��5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1��B@r��5       ��]�	X#���A� *'

learning_rate_1 t�:

loss_1�"E@�ns:5       ��]�	�M9���A� *'

learning_rate_1 t�:

loss_1@�J@U��5       ��]�	X�O���A� *'

learning_rate_1 t�:

loss_1�?@w}85       ��]�	5�e���A� *'

learning_rate_1 t�:

loss_1��;@��i5       ��]�	_|���A� *'

learning_rate_1 t�:

loss_1�?@B>,5       ��]�	bȒ���A� *'

learning_rate_1 t�:

loss_1��M@�)��5       ��]�	�
����A� *'

learning_rate_1 t�:

loss_1��D@�!�-5       ��]�	�h����A� *'

learning_rate_1 t�:

loss_1K|F@F�A5       ��]�	F�����A� *'

learning_rate_1 t�:

loss_1��?@lP�m5       ��]�	������A� *'

learning_rate_1 t�:

loss_1�1L@��5       ��]�	�&���A� *'

learning_rate_1 t�:

loss_1��8@lm�z5       ��]�	7x���A� *'

learning_rate_1 t�:

loss_1^�>@��"05       ��]�	��3���A� *'

learning_rate_1 t�:

loss_1��B@��)�5       ��]�	�K���A� *'

learning_rate_1 t�:

loss_1^�7@̙(�5       ��]�	޿a���A� *'

learning_rate_1 t�:

loss_1�YL@�(=5       ��]�	��y���A� *'

learning_rate_1 t�:

loss_1�07@!�IK5       ��]�	f2����A� *'

learning_rate_1 t�:

loss_1�)8@5j.�5       ��]�	�e����A� *'

learning_rate_1 t�:

loss_195K@�	5       ��]�	�����A� *'

learning_rate_1 t�:

loss_12�;@P��5       ��]�	`����A� *'

learning_rate_1 t�:

loss_1��M@����5       ��]�	Ϥ����A� *'

learning_rate_1 t�:

loss_1�)8@���r5       ��]�	}� ���A� *'

learning_rate_1 t�:

loss_1�N@�iߓ5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1�>A@���5       ��]�	=�-���A� *'

learning_rate_1 t�:

loss_1��E@�㊇5       ��]�	�iD���A� *'

learning_rate_1 t�:

loss_1x�N@����5       ��]�	R�Z���A� *'

learning_rate_1 t�:

loss_1��O@�z�5       ��]�	xCq���A� *'

learning_rate_1 t�:

loss_1-�H@Y#(M5       ��]�	>����A� *'

learning_rate_1 t�:

loss_1d�Z@���D5       ��]�	������A� *'

learning_rate_1 t�:

loss_1�H@3�5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1r:C@����5       ��]�	IK����A� *'

learning_rate_1 t�:

loss_1�97@De5       ��]�	������A� *'

learning_rate_1 t�:

loss_17nB@P�tF5       ��]�	1����A� *'

learning_rate_1 t�:

loss_1w�<@F��5       ��]�	����A� *'

learning_rate_1 t�:

loss_1�7@)޲�5       ��]�	��&���A� *'

learning_rate_1 t�:

loss_1�2N@��f�5       ��]�	Tl>���A� *'

learning_rate_1 t�:

loss_1�?@OS�S5       ��]�	<�T���A� *'

learning_rate_1 t�:

loss_1�t7@���Q5       ��]�	Vk���A� *'

learning_rate_1 t�:

loss_1(	K@�(5       ��]�	K����A� *'

learning_rate_1 t�:

loss_1>qG@Z¡85       ��]�	¢����A� *'

learning_rate_1 t�:

loss_16mR@aG�m5       ��]�		����A� *'

learning_rate_1 t�:

loss_1�=@a���5       ��]�	�M����A� *'

learning_rate_1 t�:

loss_1NkH@�.cx5       ��]�	����A� *'

learning_rate_1 t�:

loss_1�<E@0��5       ��]�	�����A� *'

learning_rate_1 t�:

loss_1S�?@�+��5       ��]�	�0 ���A� *'

learning_rate_1 t�:

loss_1&":@�V��5       ��]�	�� ���A� *'

learning_rate_1 t�:

loss_1)�6@�ǜ5       ��]�	�4 ���A� *'

learning_rate_1 t�:

loss_1`�L@:9�5       ��]�	�J ���A� *'

learning_rate_1 t�:

loss_1��O@0y�Y5       ��]�	��` ���A� *'

learning_rate_1 t�:

loss_1ozN@��4:5       ��]�	�Sw ���A� *'

learning_rate_1 t�:

loss_1�E>@:&��5       ��]�	��� ���A�!*'

learning_rate_1 t�:

loss_1�rC@� �75       ��]�	6ͣ ���A�!*'

learning_rate_1 t�:

loss_1wT:@)^�5       ��]�	�	� ���A�!*'

learning_rate_1 t�:

loss_1��=@fy��5       ��]�	�B� ���A�!*'

learning_rate_1 t�:

loss_1#/K@'*D�5       ��]�	/m� ���A�!*'

learning_rate_1 t�:

loss_1%�@@0�c5       ��]�	X�� ���A�!*'

learning_rate_1 t�:

loss_1w�B@?˸�5       ��]�	��!���A�!*'

learning_rate_1 t�:

loss_1IM@d8�O5       ��]�	��(!���A�!*'

learning_rate_1 t�:

loss_1C�C@��&�5       ��]�	�h@!���A�!*'

learning_rate_1 t�:

loss_1��:@��4W5       ��]�	��V!���A�!*'

learning_rate_1 t�:

loss_1�I7@��w
5       ��]�	��l!���A�!*'

learning_rate_1 t�:

loss_1VD@&���5       ��]�	�5�!���A�!*'

learning_rate_1 t�:

loss_1oI:@����5       ��]�	��!���A�!*'

learning_rate_1 t�:

loss_1�VB@Y�l5       ��]�	�7�!���A�!*'

learning_rate_1 t�:

loss_17L@,���5       ��]�	�h�!���A�!*'

learning_rate_1 t�:

loss_1�j.@T=�+5       ��]�	l�!���A�!*'

learning_rate_1 t�:

loss_1�IF@�+�5       ��]�	�;�!���A�!*'

learning_rate_1 t�:

loss_1��G@��+w5       ��]�	,�
"���A�!*'

learning_rate_1 t�:

loss_1�@@�|l�5       ��]�	�%!"���A�!*'

learning_rate_1 t�:

loss_1b:C@��L�5       ��]�	��7"���A�!*'

learning_rate_1 t�:

loss_1ؘM@�m�5       ��]�	�TM"���A�!*'

learning_rate_1 t�:

loss_1q�7@׭��5       ��]�	{�c"���A�!*'

learning_rate_1 t�:

loss_1�@@IM7&5       ��]�	��y"���A�!*'

learning_rate_1 t�:

loss_1l5D@��5       ��]�	9��"���A�!*'

learning_rate_1 t�:

loss_1�\D@����5       ��]�	S�"���A�!*'

learning_rate_1 t�:

loss_1��1@Ge�5       ��]�	��"���A�!*'

learning_rate_1 t�:

loss_1_yL@��15       ��]�	�<�"���A�!*'

learning_rate_1 t�:

loss_1�@@��2[5       ��]�	XQ�"���A�!*'

learning_rate_1 t�:

loss_1y@@W>Q5       ��]�	/e#���A�!*'

learning_rate_1 t�:

loss_1v�E@U���5       ��]�	�#���A�!*'

learning_rate_1 t�:

loss_1�=D@a0Z�5       ��]�	��/#���A�!*'

learning_rate_1 t�:

loss_1K@n�z�5       ��]�	+�E#���A�!*'

learning_rate_1 t�:

loss_1��;@���Z5       ��]�	6\#���A�!*'

learning_rate_1 t�:

loss_1j.O@�yO�5       ��]�	I0r#���A�!*'

learning_rate_1 t�:

loss_1�?H@`��T5       ��]�	.�#���A�!*'

learning_rate_1 t�:

loss_1��T@ϙq5       ��]�	���#���A�!*'

learning_rate_1 t�:

loss_1��I@wH*�5       ��]�	�,�#���A�!*'

learning_rate_1 t�:

loss_1�2@�>�5       ��]�	�M�#���A�!*'

learning_rate_1 t�:

loss_1��E@�-%�5       ��]�	3{�#���A�!*'

learning_rate_1 t�:

loss_1��4@LLhR5       ��]�	ڵ�#���A�!*'

learning_rate_1 t�:

loss_1�s/@��5       ��]�	M)$���A�!*'

learning_rate_1 t�:

loss_1��D@~�� 5       ��]�	�3($���A�!*'

learning_rate_1 t�:

loss_1��?@�X��5       ��]�	�u>$���A�!*'

learning_rate_1 t�:

loss_1Z�C@�$�p5       ��]�	��T$���A�!*'

learning_rate_1 t�:

loss_1tI@FOM�5       ��]�	qk$���A�!*'

learning_rate_1 t�:

loss_1�?N@X�W�5       ��]�	���$���A�!*'

learning_rate_1 t�:

loss_1vRI@5       ��]�	ӗ$���A�!*'

learning_rate_1 t�:

loss_1�6=@*[��5       ��]�	��$���A�!*'

learning_rate_1 t�:

loss_1�B@��y�5       ��]�	�1�$���A�!*'

learning_rate_1 t�:

loss_1��K@�g>�5       ��]�	8��$���A�!*'

learning_rate_1 t�:

loss_1�N;@:���5       ��]�	'�$���A�!*'

learning_rate_1 t�:

loss_1Z^N@���5       ��]�	��%���A�!*'

learning_rate_1 t�:

loss_1�]E@��,�5       ��]�	��%���A�!*'

learning_rate_1 t�:

loss_1��M@���5       ��]�	�5%���A�!*'

learning_rate_1 t�:

loss_1֤;@��N5       ��]�	�ML%���A�!*'

learning_rate_1 t�:

loss_1��C@�ז�5       ��]�	zsb%���A�!*'

learning_rate_1 t�:

loss_1��H@ꖩ{5       ��]�	Q�x%���A�!*'

learning_rate_1 t�:

loss_1�uK@F�q 5       ��]�	�ڎ%���A�!*'

learning_rate_1 t�:

loss_1��J@q5�5       ��]�	{��%���A�!*'

learning_rate_1 t�:

loss_100A@�#`m5       ��]�	�i�%���A�!*'

learning_rate_1 t�:

loss_10�?@�&�5       ��]�	}��%���A�!*'

learning_rate_1 t�:

loss_11*K@6_C5       ��]�	W��%���A�!*'

learning_rate_1 t�:

loss_16�A@�p�-5       ��]�	l� &���A�!*'

learning_rate_1 t�:

loss_1p�=@nMo�5       ��]�	�a&���A�!*'

learning_rate_1 t�:

loss_1�=@����5       ��]�	�/&���A�!*'

learning_rate_1 t�:

loss_1N@��5       ��]�	�F&���A�!*'

learning_rate_1 t�:

loss_1;�8@w��.5       ��]�	O\&���A�!*'

learning_rate_1 t�:

loss_1�uB@�KP�5       ��]�	�or&���A�!*'

learning_rate_1 t�:

loss_1*<@@o9�@5       ��]�	���&���A�!*'

learning_rate_1 t�:

loss_1�*V@Q:�5       ��]�	 �&���A�!*'

learning_rate_1 t�:

loss_1�>@7W L5       ��]�	��&���A�!*'

learning_rate_1 t�:

loss_1&C0@:��
5       ��]�	���&���A�!*'

learning_rate_1 t�:

loss_1�{?@l�V�5       ��]�	���&���A�!*'

learning_rate_1 t�:

loss_1�9@�`��5       ��]�	lK�&���A�!*'

learning_rate_1 t�:

loss_1��4@F.��5       ��]�	o�'���A�!*'

learning_rate_1 t�:

loss_1;�N@�P�5       ��]�	W�#'���A�!*'

learning_rate_1 t�:

loss_1#�X@39�5       ��]�	0<:'���A�!*'

learning_rate_1 t�:

loss_1k{C@�H��5       ��]�	U�P'���A�!*'

learning_rate_1 t�:

loss_1�uM@��f5       ��]�	�8g'���A�!*'

learning_rate_1 t�:

loss_1��P@+�#�5       ��]�	�~'���A�!*'

learning_rate_1 t�:

loss_1�F0@qU��5       ��]�	sY�'���A�!*'

learning_rate_1 t�:

loss_1]3>@6�&O5       ��]�	Fu�'���A�!*'

learning_rate_1 t�:

loss_1Q�F@-��5       ��]�	W��'���A�!*'

learning_rate_1 t�:

loss_1�_C@1�<5       ��]�	'b�'���A�!*'

learning_rate_1 t�:

loss_1�F@�X�M5       ��]�	`��'���A�!*'

learning_rate_1 t�:

loss_1��P@{�� 5       ��]�	��(���A�!*'

learning_rate_1 t�:

loss_1��@@�i:�5       ��]�	��(���A�!*'

learning_rate_1 t�:

loss_1ػ;@
ō�5       ��]�	YA0(���A�!*'

learning_rate_1 t�:

loss_1�pB@J�@�5       ��]�	FeF(���A�!*'

learning_rate_1 t�:

loss_1�B@̡^S5       ��]�	�\(���A�!*'

learning_rate_1 t�:

loss_14J;@u�=�5       ��]�	�r(���A�!*'

learning_rate_1 t�:

loss_1��;@���5       ��]�	5��(���A�!*'

learning_rate_1 t�:

loss_1��0@~&�*5       ��]�	,�(���A�!*'

learning_rate_1 t�:

loss_1��?@��	R5       ��]�	i �(���A�!*'

learning_rate_1 t�:

loss_1��T@��R5       ��]�	�Y�(���A�!*'

learning_rate_1 t�:

loss_1Z�H@�}1�5       ��]�	�T�(���A�!*'

learning_rate_1 t�:

loss_1�S?@��5       ��]�	���(���A�!*'

learning_rate_1 t�:

loss_158D@�G-�5       ��]�	��)���A�!*'

learning_rate_1 t�:

loss_1
>@Z��5       ��]�	�&)���A�!*'

learning_rate_1 t�:

loss_1��D@��Y,5       ��]�	=)���A�!*'

learning_rate_1 t�:

loss_1��C@ѩ�5       ��]�	�tS)���A�!*'

learning_rate_1 t�:

loss_1]P@�i��5       ��]�	dpk)���A�!*'

learning_rate_1 t�:

loss_1��>@+��5       ��]�	��)���A�!*'

learning_rate_1 t�:

loss_1:L6@��5       ��]�	p��)���A�!*'

learning_rate_1 t�:

loss_1�C@B�t5       ��]�	�ͮ)���A�!*'

learning_rate_1 t�:

loss_1ͬ?@w0i�5       ��]�	f�)���A�!*'

learning_rate_1 t�:

loss_1Y=@c�	�5       ��]�	���)���A�!*'

learning_rate_1 t�:

loss_18HE@���5       ��]�	���)���A�!*'

learning_rate_1 t�:

loss_1�8@h�E�5       ��]�	��*���A�!*'

learning_rate_1 t�:

loss_1�H@۪E�5       ��]�	n�*���A�!*'

learning_rate_1 t�:

loss_1�F@���5       ��]�	�6*���A�!*'

learning_rate_1 t�:

loss_1�0D@:��5       ��]�	0�L*���A�!*'

learning_rate_1 t�:

loss_1��<@u{D5       ��]�	��b*���A�!*'

learning_rate_1 t�:

loss_1�R@@j]�5       ��]�	y*���A�!*'

learning_rate_1 t�:

loss_1=6@ߝ15       ��]�	�k�*���A�!*'

learning_rate_1 t�:

loss_1��K@�x�5       ��]�	I��*���A�!*'

learning_rate_1 t�:

loss_1�B@ru��5       ��]�	x��*���A�!*'

learning_rate_1 t�:

loss_1�)B@h)p5       ��]�	�Q�*���A�!*'

learning_rate_1 t�:

loss_1Z"?@`�qd5       ��]�	���*���A�!*'

learning_rate_1 t�:

loss_1��7@ /]5       ��]�	�a�*���A�!*'

learning_rate_1 t�:

loss_1�$I@C�[5       ��]�	�w+���A�!*'

learning_rate_1 t�:

loss_14�O@冮�5       ��]�	:`.+���A�!*'

learning_rate_1 t�:

loss_1��;@�V5       ��]�	��D+���A�!*'

learning_rate_1 t�:

loss_1#vI@��_5       ��]�	x�Z+���A�!*'

learning_rate_1 t�:

loss_1R!>@��1l5       ��]�	��n+���A�!*'

learning_rate_1 t�:

loss_1�.R@	o�5       ��]�	���+���A�!*'

learning_rate_1 t�:

loss_1�8@e4!u5       ��]�	� �+���A�!*'

learning_rate_1 t�:

loss_1��=@3ɔ5       ��]�	T+�+���A�!*'

learning_rate_1 t�:

loss_1p�C@L[��5       ��]�	
��+���A�"*'

learning_rate_1 t�:

loss_1��:@�6�y5       ��]�	��+���A�"*'

learning_rate_1 t�:

loss_1�ED@ص5       ��]�	~�+���A�"*'

learning_rate_1 t�:

loss_1<�9@S^�5       ��]�	�,���A�"*'

learning_rate_1 t�:

loss_1m�;@S�_�5       ��]�	�d#,���A�"*'

learning_rate_1 t�:

loss_1=F@/�5       ��]�	�9,���A�"*'

learning_rate_1 t�:

loss_1m?9@��5       ��]�	�O,���A�"*'

learning_rate_1 t�:

loss_1�NS@9"��5       ��]�		�e,���A�"*'

learning_rate_1 t�:

loss_1��@@j1i�5       ��]�	NK|,���A�"*'

learning_rate_1 t�:

loss_1�OJ@'t5       ��]�	�ړ,���A�"*'

learning_rate_1 t�:

loss_1�8@�{\5       ��]�	 �,���A�"*'

learning_rate_1 t�:

loss_1~�:@���5       ��]�	zx�,���A�"*'

learning_rate_1 t�:

loss_1{K@�T�5       ��]�	���,���A�"*'

learning_rate_1 t�:

loss_1]�I@�X��5       ��]�	]��,���A�"*'

learning_rate_1 t�:

loss_1_�A@>��5       ��]�	�-���A�"*'

learning_rate_1 t�:

loss_1
�K@���+5       ��]�	��-���A�"*'

learning_rate_1 t�:

loss_1�<C@���5       ��]�	�Q0-���A�"*'

learning_rate_1 t�:

loss_1��7@l(<5       ��]�	��F-���A�"*'

learning_rate_1 t�:

loss_1m�7@Z�,�5       ��]�	�)]-���A�"*'

learning_rate_1 t�:

loss_1��E@G��5       ��]�	��t-���A�"*'

learning_rate_1 t�:

loss_1�{0@����5       ��]�	K!�-���A�"*'

learning_rate_1 t�:

loss_1�K@��5       ��]�	�X�-���A�"*'

learning_rate_1 t�:

loss_1Bx>@=t�(5       ��]�	YL�-���A�"*'

learning_rate_1 t�:

loss_1[CB@�)h�5       ��]�	�g�-���A�"*'

learning_rate_1 t�:

loss_1�0I@�o�55       ��]�	�I�-���A�"*'

learning_rate_1 t�:

loss_1��L@�ce5       ��]�	z��-���A�"*'

learning_rate_1 t�:

loss_1Q0;@=�.�5       ��]�	_�.���A�"*'

learning_rate_1 t�:

loss_1�U@@����5       ��]�	x'.���A�"*'

learning_rate_1 t�:

loss_1I�N@��-5       ��]�	��=.���A�"*'

learning_rate_1 t�:

loss_1�:@�z�5       ��]�	UjT.���A�"*'

learning_rate_1 t�:

loss_1��0@Q9�5       ��]�	��j.���A�"*'

learning_rate_1 t�:

loss_1��:@U0�g5       ��]�	q��.���A�"*'

learning_rate_1 t�:

loss_1Mg@@�;u�5       ��]�	�C�.���A�"*'

learning_rate_1 t�:

loss_1��E@��|�5       ��]�	�.���A�"*'

learning_rate_1 t�:

loss_1��H@���l5       ��]�	{��.���A�"*'

learning_rate_1 t�:

loss_1*�=@d��\5       ��]�	���.���A�"*'

learning_rate_1 t�:

loss_1��E@�-�X5       ��]�	���.���A�"*'

learning_rate_1 t�:

loss_1��D@Z�e5       ��]�	�%/���A�"*'

learning_rate_1 t�:

loss_1LqC@�9�T5       ��]�	#m/���A�"*'

learning_rate_1 t�:

loss_1�f?@:n9�5       ��]�	��4/���A�"*'

learning_rate_1 t�:

loss_1F�5@��|5       ��]�	;'K/���A�"*'

learning_rate_1 t�:

loss_1-#M@	¡5       ��]�	\�a/���A�"*'

learning_rate_1 t�:

loss_1΋>@�?�5       ��]�	��w/���A�"*'

learning_rate_1 t�:

loss_1G@�K�5       ��]�	�D�/���A�"*'

learning_rate_1 t�:

loss_1�:I@f��F5       ��]�	���/���A�"*'

learning_rate_1 t�:

loss_1�lG@��|5       ��]�	[�/���A�"*'

learning_rate_1 t�:

loss_1/�B@5<�^5       ��]�	��/���A�"*'

learning_rate_1 t�:

loss_1`4@��o5       ��]�	��/���A�"*'

learning_rate_1 t�:

loss_1n`G@t�T5       ��]�	C(�/���A�"*'

learning_rate_1 t�:

loss_1��:@��.[5       ��]�	��0���A�"*'

learning_rate_1 t�:

loss_1%<>@�[5       ��]�	"+0���A�"*'

learning_rate_1 t�:

loss_12�;@��B5       ��]�	9XA0���A�"*'

learning_rate_1 t�:

loss_1�5@%�\~5       ��]�	k�W0���A�"*'

learning_rate_1 t�:

loss_1�L@Q~�h5       ��]�	��m0���A�"*'

learning_rate_1 t�:

loss_1$^T@G��05       ��]�		@�0���A�"*'

learning_rate_1 t�:

loss_1S<@좤�5       ��]�	{e�0���A�"*'

learning_rate_1 t�:

loss_1�uC@�vU5       ��]�	R$�0���A�"*'

learning_rate_1 t�:

loss_1�VH@��5       ��]�	P��0���A�"*'

learning_rate_1 t�:

loss_1{�C@�'�C5       ��]�	���0���A�"*'

learning_rate_1 t�:

loss_1��C@"���5       ��]�	]]�0���A�"*'

learning_rate_1 t�:

loss_1� ?@�G��5       ��]�	��
1���A�"*'

learning_rate_1 t�:

loss_1��:@Z�+5       ��]�	�!1���A�"*'

learning_rate_1 t�:

loss_1�2F@��E�5       ��]�	�71���A�"*'

learning_rate_1 t�:

loss_1�^A@��s_5       ��]�	OM1���A�"*'

learning_rate_1 t�:

loss_1s�B@2�ҳ5       ��]�	�c1���A�"*'

learning_rate_1 t�:

loss_1<}D@� V�5       ��]�	��y1���A�"*'

learning_rate_1 t�:

loss_1�E@>l�5       ��]�	#�1���A�"*'

learning_rate_1 t�:

loss_1
@F@���5       ��]�	]k�1���A�"*'

learning_rate_1 t�:

loss_1��O@�_�5       ��]�	�r�1���A�"*'

learning_rate_1 t�:

loss_1ܫH@̼<�5       ��]�	A��1���A�"*'

learning_rate_1 t�:

loss_1�NC@���Q5       ��]�	��1���A�"*'

learning_rate_1 t�:

loss_1�D@�tWt5       ��]�	�O�1���A�"*'

learning_rate_1 t�:

loss_1k`E@�"�5       ��]�	�42���A�"*'

learning_rate_1 t�:

loss_1� @@m.O�5       ��]�	�V-2���A�"*'

learning_rate_1 t�:

loss_1
�:@�!ے5       ��]�	ʁC2���A�"*'

learning_rate_1 t�:

loss_1@�G@����5       ��]�	;�Y2���A�"*'

learning_rate_1 t�:

loss_1?�<@'�ә5       ��]�	��o2���A�"*'

learning_rate_1 t�:

loss_1��M@+�g5       ��]�	?�2���A�"*'

learning_rate_1 t�:

loss_1�/1@5.M5       ��]�	l��2���A�"*'

learning_rate_1 t�:

loss_1E@Hǜ�5       ��]�	?�2���A�"*'

learning_rate_1 t�:

loss_1�=@��ٜ5       ��]�	�C�2���A�"*'

learning_rate_1 t�:

loss_10�F@�x��5       ��]�	�a�2���A�"*'

learning_rate_1 t�:

loss_1�iR@ T/5       ��]�	c�2���A�"*'

learning_rate_1 t�:

loss_1 �:@�1�S5       ��]�	�3���A�"*'

learning_rate_1 t�:

loss_1�=>@i�w5       ��]�	C�!3���A�"*'

learning_rate_1 t�:

loss_1xJ@7�75       ��]�	1;83���A�"*'

learning_rate_1 t�:

loss_1��D@dM3[5       ��]�	�pN3���A�"*'

learning_rate_1 t�:

loss_1/>@c+ �5       ��]�	b�e3���A�"*'

learning_rate_1 t�:

loss_1Y�;@����5       ��]�	]|3���A�"*'

learning_rate_1 t�:

loss_1]#D@k��'5       ��]�	w��3���A�"*'

learning_rate_1 t�:

loss_1�"H@|SM$5       ��]�	ږ�3���A�"*'

learning_rate_1 t�:

loss_1��@@��n25       ��]�	���3���A�"*'

learning_rate_1 t�:

loss_1��I@t�5       ��]�	22�3���A�"*'

learning_rate_1 t�:

loss_1�,@o9�5       ��]�	���3���A�"*'

learning_rate_1 t�:

loss_1ԸA@{ђ55       ��]�	�f4���A�"*'

learning_rate_1 t�:

loss_1=�7@)��p5       ��]�	ݿ4���A�"*'

learning_rate_1 t�:

loss_1��5@��T�5       ��]�	�24���A�"*'

learning_rate_1 t�:

loss_1�#:@��f5       ��]�	��H4���A�"*'

learning_rate_1 t�:

loss_10�5@`�X5       ��]�	��^4���A�"*'

learning_rate_1 t�:

loss_1u,M@|�2�5       ��]�	08u4���A�"*'

learning_rate_1 t�:

loss_1j^A@Wl�5       ��]�		��4���A�"*'

learning_rate_1 t�:

loss_1�F@�Ƽ�5       ��]�	��4���A�"*'

learning_rate_1 t�:

loss_1#�;@�P~�5       ��]�	QQ�4���A�"*'

learning_rate_1 t�:

loss_1�gP@�K�5       ��]�	=��4���A�"*'

learning_rate_1 t�:

loss_1�K@�Az5       ��]�	���4���A�"*'

learning_rate_1 t�:

loss_1��?@0���5       ��]�	5��4���A�"*'

learning_rate_1 t�:

loss_1��:@¿��5       ��]�	"z5���A�"*'

learning_rate_1 t�:

loss_1iZ8@�� �5       ��]�	�'5���A�"*'

learning_rate_1 t�:

loss_1��;@<���5       ��]�	�Z>5���A�"*'

learning_rate_1 t�:

loss_1K�M@c:F5       ��]�	"pT5���A�"*'

learning_rate_1 t�:

loss_1zd@@�O]5       ��]�	&�j5���A�"*'

learning_rate_1 t�:

loss_1piO@���#5       ��]�	X�5���A�"*'

learning_rate_1 t�:

loss_1D�4@�/��5       ��]�	O��5���A�"*'

learning_rate_1 t�:

loss_1��@@(�805       ��]�	k��5���A�"*'

learning_rate_1 t�:

loss_1��N@��K 5       ��]�	"��5���A�"*'

learning_rate_1 t�:

loss_1�9@���5       ��]�	4��5���A�"*'

learning_rate_1 t�:

loss_1�B@�ח5       ��]�	]�5���A�"*'

learning_rate_1 t�:

loss_1��F@�f�5       ��]�	C:6���A�"*'

learning_rate_1 t�:

loss_1D@j/�g5       ��]�	�o6���A�"*'

learning_rate_1 t�:

loss_1�TP@�.jt5       ��]�	@�46���A�"*'

learning_rate_1 t�:

loss_1.�D@R��5       ��]�	f�J6���A�"*'

learning_rate_1 t�:

loss_1_�G@��\�5       ��]�	�Yb6���A�"*'

learning_rate_1 t�:

loss_1��7@b�5       ��]�	+�x6���A�"*'

learning_rate_1 t�:

loss_1P0C@�svz5       ��]�	uj�6���A�"*'

learning_rate_1 t�:

loss_1}�G@]r%�5       ��]�	�¥6���A�"*'

learning_rate_1 t�:

loss_1��O@�L��5       ��]�	d�6���A�"*'

learning_rate_1 t�:

loss_1�X7@��d�5       ��]�	�A�6���A�"*'

learning_rate_1 t�:

loss_1ئI@��Q5       ��]�	e��6���A�"*'

learning_rate_1 t�:

loss_1m;C@
��)5       ��]�	�&�6���A�#*'

learning_rate_1 t�:

loss_1�E@gH�5       ��]�	�7���A�#*'

learning_rate_1 t�:

loss_1ϩD@ak�5       ��]�	��+7���A�#*'

learning_rate_1 t�:

loss_1_nJ@JqQa5       ��]�	�xB7���A�#*'

learning_rate_1 t�:

loss_1�I@9a�25       ��]�	BY7���A�#*'

learning_rate_1 t�:

loss_1L�A@��805       ��]�	8�o7���A�#*'

learning_rate_1 t�:

loss_1�^?@�%�)5       ��]�	bb�7���A�#*'

learning_rate_1 t�:

loss_1N`B@�Gp5       ��]�	���7���A�#*'

learning_rate_1 t�:

loss_1V�N@��w�5       ��]�	T��7���A�#*'

learning_rate_1 t�:

loss_1�nM@��(5       ��]�	�'�7���A�#*'

learning_rate_1 t�:

loss_1&�L@�8�;5       ��]�	k�7���A�#*'

learning_rate_1 t�:

loss_1/l8@�L@5       ��]�	��7���A�#*'

learning_rate_1 t�:

loss_1 &C@h��@5       ��]�	�8���A�#*'

learning_rate_1 t�:

loss_1<;<@�Q5       ��]�	�"8���A�#*'

learning_rate_1 t�:

loss_1~�A@�6�5       ��]�	�798���A�#*'

learning_rate_1 t�:

loss_1�I@$�(�5       ��]�	[�O8���A�#*'

learning_rate_1 t�:

loss_1#-Z@��)5       ��]�	��e8���A�#*'

learning_rate_1 t�:

loss_1$�<@@�$�5       ��]�	�|8���A�#*'

learning_rate_1 t�:

loss_1��:@��ID5       ��]�	2�8���A�#*'

learning_rate_1 t�:

loss_1��E@e�2*5       ��]�	��8���A�#*'

learning_rate_1 t�:

loss_1��2@_���5       ��]�	v<�8���A�#*'

learning_rate_1 t�:

loss_1c�E@�lv�5       ��]�	���8���A�#*'

learning_rate_1 t�:

loss_1��D@�C�5       ��]�	�8�8���A�#*'

learning_rate_1 t�:

loss_1^n?@�-��5       ��]�	 �9���A�#*'

learning_rate_1 t�:

loss_1m�;@`�?5       ��]�	7�9���A�#*'

learning_rate_1 t�:

loss_1a�J@�Ϲd5       ��]�	>�09���A�#*'

learning_rate_1 t�:

loss_1spE@|�\5       ��]�	(G9���A�#*'

learning_rate_1 t�:

loss_1�k?@j�_�5       ��]�	�Q]9���A�#*'

learning_rate_1 t�:

loss_1�@@j�e�5       ��]�	puu9���A�#*'

learning_rate_1 t�:

loss_1��2@L)Z5       ��]�	��9���A�#*'

learning_rate_1 t�:

loss_1,�?@�� 5       ��]�	-�9���A�#*'

learning_rate_1 t�:

loss_1Q�E@yF�5       ��]�	:U�9���A�#*'

learning_rate_1 t�:

loss_1-�,@fL>�5       ��]�	�c�9���A�#*'

learning_rate_1 t�:

loss_1��[@5˨5       ��]�	_��9���A�#*'

learning_rate_1 t�:

loss_1w�M@>��5       ��]�	��9���A�#*'

learning_rate_1 t�:

loss_17�?@mHHf5       ��]�	�5:���A�#*'

learning_rate_1 t�:

loss_1�G@�wF5       ��]�	��*:���A�#*'

learning_rate_1 t�:

loss_1}�D@1j�I5       ��]�	_�@:���A�#*'

learning_rate_1 t�:

loss_1�G@��V�5       ��]�	��V:���A�#*'

learning_rate_1 t�:

loss_1��H@�u�5       ��]�	�?m:���A�#*'

learning_rate_1 t�:

loss_1��A@��y�5       ��]�	���:���A�#*'

learning_rate_1 t�:

loss_1?�U@KqY5       ��]�	�[�:���A�#*'

learning_rate_1 t�:

loss_1��@@��ӽ5       ��]�	�ֱ:���A�#*'

learning_rate_1 t�:

loss_1Ƒ1@lݪ5       ��]�	���:���A�#*'

learning_rate_1 t�:

loss_1X6G@&�d5       ��]�	UI�:���A�#*'

learning_rate_1 t�:

loss_16�)@�_�5       ��]�	v��:���A�#*'

learning_rate_1 t�:

loss_18|;@-�5       ��]�	Nj;���A�#*'

learning_rate_1 t�:

loss_1�?S@�}�5       ��]�	8�$;���A�#*'

learning_rate_1 t�:

loss_1�J?@�Q�5       ��]�	7I<;���A�#*'

learning_rate_1 t�:

loss_1��=@/���5       ��]�	��R;���A�#*'

learning_rate_1 t�:

loss_1�1:@���V5       ��]�	}�h;���A�#*'

learning_rate_1 t�:

loss_1��@@�yD�5       ��]�	�];���A�#*'

learning_rate_1 t�:

loss_1\�N@/�Y5       ��]�	Ϭ�;���A�#*'

learning_rate_1 t�:

loss_1{�/@�Ik5       ��]�	׫;���A�#*'

learning_rate_1 t�:

loss_1��D@3�?D5       ��]�	m��;���A�#*'

learning_rate_1 t�:

loss_1�2F@��X�5       ��]�	�;���A�#*'

learning_rate_1 t�:

loss_1 8@�J�5       ��]�	,0�;���A�#*'

learning_rate_1 t�:

loss_1��J@�kǏ5       ��]�	��<���A�#*'

learning_rate_1 t�:

loss_1�6@����5       ��]�	��<���A�#*'

learning_rate_1 t�:

loss_1�e?@Z�b�5       ��]�	��1<���A�#*'

learning_rate_1 t�:

loss_1C�H@XS�15       ��]�	�[H<���A�#*'

learning_rate_1 t�:

loss_15�V@J���5       ��]�	�Uf<���A�#*'

learning_rate_1>J�:

loss_1�F8@�[5       ��]�	�|<���A�#*'

learning_rate_1>J�:

loss_1��D@�vr�5       ��]�	�0�<���A�#*'

learning_rate_1>J�:

loss_1�@@�:}�5       ��]�	4~�<���A�#*'

learning_rate_1>J�:

loss_1�Q@ք��5       ��]�	�ѿ<���A�#*'

learning_rate_1>J�:

loss_1�:@�B��5       ��]�	@��<���A�#*'

learning_rate_1>J�:

loss_1>cA@�N݀5       ��]�	H��<���A�#*'

learning_rate_1>J�:

loss_1]:@@���_5       ��]�	l$=���A�#*'

learning_rate_1>J�:

loss_1��K@�Oe5       ��]�	�=���A�#*'

learning_rate_1>J�:

loss_1��G@ tq�5       ��]�	>/=���A�#*'

learning_rate_1>J�:

loss_19�L@��5       ��]�	�{E=���A�#*'

learning_rate_1>J�:

loss_1�:;@���5       ��]�	�	\=���A�#*'

learning_rate_1>J�:

loss_1+�B@���5       ��]�	�0t=���A�#*'

learning_rate_1>J�:

loss_1V�/@� [5       ��]�	�h�=���A�#*'

learning_rate_1>J�:

loss_1��7@j��45       ��]�	_��=���A�#*'

learning_rate_1>J�:

loss_1�{I@�"�5       ��]�	*@�=���A�#*'

learning_rate_1>J�:

loss_1��+@�i5       ��]�	�#�=���A�#*'

learning_rate_1>J�:

loss_1s�C@�7��5       ��]�	X�=���A�#*'

learning_rate_1>J�:

loss_1��D@����5       ��]�	6<�=���A�#*'

learning_rate_1>J�:

loss_1P�5@��#�5       ��]�	�~>���A�#*'

learning_rate_1>J�:

loss_1c~=@�\N5       ��]�	UU)>���A�#*'

learning_rate_1>J�:

loss_1��B@=�Ո5       ��]�	��?>���A�#*'

learning_rate_1>J�:

loss_1{7@��E:5       ��]�	�U>���A�#*'

learning_rate_1>J�:

loss_1-/;@5��5       ��]�	�}l>���A�#*'

learning_rate_1>J�:

loss_1�E@ٳ�5       ��]�	��>���A�#*'

learning_rate_1>J�:

loss_1	�5@�˺�5       ��]�	<��>���A�#*'

learning_rate_1>J�:

loss_1_5@�cs�5       ��]�	��>���A�#*'

learning_rate_1>J�:

loss_1|B@��|�5       ��]�	e]�>���A�#*'

learning_rate_1>J�:

loss_1
�>@`w��5       ��]�	h��>���A�#*'

learning_rate_1>J�:

loss_1�jA@���5       ��]�	i��>���A�#*'

learning_rate_1>J�:

loss_1�6I@��N�5       ��]�	�
?���A�#*'

learning_rate_1>J�:

loss_1��C@q_��5       ��]�	S ?���A�#*'

learning_rate_1>J�:

loss_1��:@0M��5       ��]�	1Q7?���A�#*'

learning_rate_1>J�:

loss_1�#0@2:�5       ��]�	�M?���A�#*'

learning_rate_1>J�:

loss_1�ZC@�P��5       ��]�	��e?���A�#*'

learning_rate_1>J�:

loss_1��-@�ϫ5       ��]�	� |?���A�#*'

learning_rate_1>J�:

loss_17@�8�/5       ��]�	=;�?���A�#*'

learning_rate_1>J�:

loss_1�`D@.��25       ��]�	���?���A�#*'

learning_rate_1>J�:

loss_1�;@���k5       ��]�	/��?���A�#*'

learning_rate_1>J�:

loss_1׺P@f��s5       ��]�	9��?���A�#*'

learning_rate_1>J�:

loss_1�n4@�+�5       ��]�	���?���A�#*'

learning_rate_1>J�:

loss_10�C@��u5       ��]�	o# @���A�#*'

learning_rate_1>J�:

loss_12�@@�b�R5       ��]�	J@���A�#*'

learning_rate_1>J�:

loss_1��B@"/K5       ��]�	�L-@���A�#*'

learning_rate_1>J�:

loss_1-g=@���5       ��]�	F�C@���A�#*'

learning_rate_1>J�:

loss_1f=6@/�@5       ��]�	#�Y@���A�#*'

learning_rate_1>J�:

loss_1�:@%�i5       ��]�	�p@���A�#*'

learning_rate_1>J�:

loss_1*�@@�=.�5       ��]�	��@���A�#*'

learning_rate_1>J�:

loss_1=�?@N)w
5       ��]�	�c�@���A�#*'

learning_rate_1>J�:

loss_1��G@��}�5       ��]�	O��@���A�#*'

learning_rate_1>J�:

loss_174@6G�5       ��]�	Kh�@���A�#*'

learning_rate_1>J�:

loss_1��8@���5       ��]�	���@���A�#*'

learning_rate_1>J�:

loss_1Կ;@���5       ��]�	�<�@���A�#*'

learning_rate_1>J�:

loss_1U>Q@�4�x5       ��]�	�A���A�#*'

learning_rate_1>J�:

loss_1y<@���L5       ��]�	��"A���A�#*'

learning_rate_1>J�:

loss_1�@@p(��5       ��]�	��8A���A�#*'

learning_rate_1>J�:

loss_1�":@]�V5       ��]�	�mOA���A�#*'

learning_rate_1>J�:

loss_1"3@@���5       ��]�	��eA���A�#*'

learning_rate_1>J�:

loss_1�V<@,�-H5       ��]�	V�{A���A�#*'

learning_rate_1>J�:

loss_1ppH@_[�5       ��]�	�G�A���A�#*'

learning_rate_1>J�:

loss_1JjF@u��b5       ��]�	-éA���A�#*'

learning_rate_1>J�:

loss_15�2@S#G"5       ��]�	c�A���A�#*'

learning_rate_1>J�:

loss_1��L@�+.�5       ��]�	.b�A���A�#*'

learning_rate_1>J�:

loss_1��P@�P0�5       ��]�	/��A���A�#*'

learning_rate_1>J�:

loss_1$�=@�<[5       ��]�	}B���A�#*'

learning_rate_1>J�:

loss_1#\C@�s��5       ��]�	�AB���A�#*'

learning_rate_1>J�:

loss_1��H@�K~\5       ��]�	�P/B���A�#*'

learning_rate_1>J�:

loss_1�=@�?�5       ��]�	��EB���A�#*'

learning_rate_1>J�:

loss_1|�M@A��5       ��]�	~�[B���A�#*'

learning_rate_1>J�:

loss_1̲W@��N5       ��]�	SrB���A�#*'

learning_rate_1>J�:

loss_1��?@U&"]5       ��]�	�Z�B���A�#*'

learning_rate_1>J�:

loss_1Z1@�xb5       ��]�	*��B���A�#*'

learning_rate_1>J�:

loss_1)MO@i �35       ��]�	�'�B���A�#*'

learning_rate_1>J�:

loss_12P<@sٴZ5       ��]�	���B���A�#*'

learning_rate_1>J�:

loss_1�c5@]z�5       ��]�	���B���A�#*'

learning_rate_1>J�:

loss_1=@���95       ��]�	���B���A�#*'

learning_rate_1>J�:

loss_1�C@Ը;5       ��]�	�C���A�#*'

learning_rate_1>J�:

loss_1ي@@7�Ψ5       ��]�	�:%C���A�#*'

learning_rate_1>J�:

loss_1��<@��Z�5       ��]�	O<C���A�#*'

learning_rate_1>J�:

loss_1��C@��xK5       ��]�	�RC���A�#*'

learning_rate_1>J�:

loss_1��?@e|��5       ��]�	0�hC���A�#*'

learning_rate_1>J�:

loss_1�KC@pJs"5       ��]�	�:C���A�#*'

learning_rate_1>J�:

loss_1��2@g�B�5       ��]�	*��C���A�#*'

learning_rate_1>J�:

loss_1�5?@�w�|5       ��]�	U�C���A�#*'

learning_rate_1>J�:

loss_1��<@ap�5       ��]�	X��C���A�#*'

learning_rate_1>J�:

loss_1=u.@���5       ��]�	���C���A�#*'

learning_rate_1>J�:

loss_1�fB@�.>�5       ��]�	���C���A�#*'

learning_rate_1>J�:

loss_1�<@����5       ��]�	��D���A�#*'

learning_rate_1>J�:

loss_1x 6@�@�=5       ��]�	��D���A�#*'

learning_rate_1>J�:

loss_13�P@ꄲ_5       ��]�	h�2D���A�#*'

learning_rate_1>J�:

loss_1͚G@�< 5       ��]�	i�ID���A�#*'

learning_rate_1>J�:

loss_19�@@��(_5       ��]�	}�`D���A�#*'

learning_rate_1>J�:

loss_1��C@�Y�`5       ��]�	�+wD���A�#*'

learning_rate_1>J�:

loss_1A5@ǼRV5       ��]�	�y�D���A�#*'

learning_rate_1>J�:

loss_1�A@����5       ��]�	
��D���A�#*'

learning_rate_1>J�:

loss_1�>@F1�5       ��]�	t�D���A�#*'

learning_rate_1>J�:

loss_1�a9@븻5       ��]�	W��D���A�#*'

learning_rate_1>J�:

loss_1�/E@1�5       ��]�	�%�D���A�#*'

learning_rate_1>J�:

loss_1��?@b9t�5       ��]�	 ��D���A�#*'

learning_rate_1>J�:

loss_1�=@�~5       ��]�	�BE���A�#*'

learning_rate_1>J�:

loss_1��C@o(	�5       ��]�	�I+E���A�#*'

learning_rate_1>J�:

loss_1Ģ3@�{#v5       ��]�	ҢAE���A�#*'

learning_rate_1>J�:

loss_1�8@����5       ��]�	�WE���A�#*'

learning_rate_1>J�:

loss_1'�>@R��5       ��]�	��oE���A�#*'

learning_rate_1>J�:

loss_1�c0@�S�@5       ��]�	�l�E���A�#*'

learning_rate_1>J�:

loss_1��>@QS5       ��]�	���E���A�#*'

learning_rate_1>J�:

loss_1��>@&���5       ��]�	�E���A�#*'

learning_rate_1>J�:

loss_1<�>@ى��5       ��]�	���E���A�#*'

learning_rate_1>J�:

loss_1!�:@���5       ��]�	��E���A�$*'

learning_rate_1>J�:

loss_1�F@��5       ��]�	1��E���A�$*'

learning_rate_1>J�:

loss_1�
J@=��5       ��]�	�+F���A�$*'

learning_rate_1>J�:

loss_10)?@({tp5       ��]�	n�#F���A�$*'

learning_rate_1>J�:

loss_1]5@��<X5       ��]�	��:F���A�$*'

learning_rate_1>J�:

loss_1��D@+5       ��]�	�RF���A�$*'

learning_rate_1>J�:

loss_1��4@o|�	5       ��]�	��hF���A�$*'

learning_rate_1>J�:

loss_1��B@VEz�5       ��]�	�F���A�$*'

learning_rate_1>J�:

loss_1<E@O���5       ��]�	\וF���A�$*'

learning_rate_1>J�:

loss_1l}<@�(?�5       ��]�	���F���A�$*'

learning_rate_1>J�:

loss_1l�=@ͽ�5       ��]�	�+�F���A�$*'

learning_rate_1>J�:

loss_1
�7@'��5       ��]�	�_�F���A�$*'

learning_rate_1>J�:

loss_1��J@�#�5       ��]�	w��F���A�$*'

learning_rate_1>J�:

loss_1�@@@kw�5       ��]�	JG���A�$*'

learning_rate_1>J�:

loss_1d�F@�>�`5       ��]�	>XG���A�$*'

learning_rate_1>J�:

loss_1��6@
�5       ��]�	s23G���A�$*'

learning_rate_1>J�:

loss_1>�U@ɼ�5       ��]�	�kIG���A�$*'

learning_rate_1>J�:

loss_1Fs@@����5       ��]�	��_G���A�$*'

learning_rate_1>J�:

loss_1ZM@���"5       ��]�	.�uG���A�$*'

learning_rate_1>J�:

loss_1�7R@v�o5       ��]�	όG���A�$*'

learning_rate_1>J�:

loss_11B@x�	5       ��]�	0�G���A�$*'

learning_rate_1>J�:

loss_1��D@1Z�5       ��]�	�Y�G���A�$*'

learning_rate_1>J�:

loss_1GtG@]�+5       ��]�	���G���A�$*'

learning_rate_1>J�:

loss_1��>@���5       ��]�	���G���A�$*'

learning_rate_1>J�:

loss_1aC@ �	�5       ��]�	�I�G���A�$*'

learning_rate_1>J�:

loss_1��8@wo�/5       ��]�	�+H���A�$*'

learning_rate_1>J�:

loss_15E@M�1 5       ��]�	G�(H���A�$*'

learning_rate_1>J�:

loss_1~9G@ ]�5       ��]�	3'?H���A�$*'

learning_rate_1>J�:

loss_1cqO@�|5       ��]�	�zUH���A�$*'

learning_rate_1>J�:

loss_1�;@\��5       ��]�	��kH���A�$*'

learning_rate_1>J�:

loss_1��E@.g��5       ��]�	O:�H���A�$*'

learning_rate_1>J�:

loss_1ʲC@��#5       ��]�	�d�H���A�$*'

learning_rate_1>J�:

loss_1�nE@�F�5       ��]�	^��H���A�$*'

learning_rate_1>J�:

loss_1�t8@y��J5       ��]�	B��H���A�$*'

learning_rate_1>J�:

loss_1��B@O<_5       ��]�	���H���A�$*'

learning_rate_1>J�:

loss_1
<@dSd�5       ��]�	�'�H���A�$*'

learning_rate_1>J�:

loss_1��G@x1h�5       ��]�	4�I���A�$*'

learning_rate_1>J�:

loss_1��?@�h�F5       ��]�	�I���A�$*'

learning_rate_1>J�:

loss_1�!G@�3X5       ��]�	�4I���A�$*'

learning_rate_1>J�:

loss_1i�@@d�c5       ��]�	*vJI���A�$*'

learning_rate_1>J�:

loss_1Q@@����5       ��]�	��`I���A�$*'

learning_rate_1>J�:

loss_1X�[@�#��5       ��]�	��vI���A�$*'

learning_rate_1>J�:

loss_1z�?@���5       ��]�	4�I���A�$*'

learning_rate_1>J�:

loss_1�7@�S	�5       ��]�	B��I���A�$*'

learning_rate_1>J�:

loss_10�?@�k�05       ��]�	*ڻI���A�$*'

learning_rate_1>J�:

loss_1֛7@22�G5       ��]�	'��I���A�$*'

learning_rate_1>J�:

loss_1K�F@�f;X5       ��]�	w��I���A�$*'

learning_rate_1>J�:

loss_1}0@,s�l5       ��]�	"� J���A�$*'

learning_rate_1>J�:

loss_1v�7@��ȳ5       ��]�	��J���A�$*'

learning_rate_1>J�:

loss_1^7A@h��5       ��]�	H/J���A�$*'

learning_rate_1>J�:

loss_1,�5@ϐ��5       ��]�	m�FJ���A�$*'

learning_rate_1>J�:

loss_1�)I@U
>�5       ��]�	�^]J���A�$*'

learning_rate_1>J�:

loss_1BSF@ �ź5       ��]�	��sJ���A�$*'

learning_rate_1>J�:

loss_1t�B@_�X�5       ��]�	��J���A�$*'

learning_rate_1>J�:

loss_1mF@u�5       ��]�	á�J���A�$*'

learning_rate_1>J�:

loss_1?@!M(5       ��]�	{��J���A�$*'

learning_rate_1>J�:

loss_1�:@}��p5       ��]�	I-�J���A�$*'

learning_rate_1>J�:

loss_1	IE@UcƸ5       ��]�	|V�J���A�$*'

learning_rate_1>J�:

loss_1	�5@mK�5       ��]�	��J���A�$*'

learning_rate_1>J�:

loss_1 ZB@�&'�5       ��]�	7�K���A�$*'

learning_rate_1>J�:

loss_1NdF@*w`�5       ��]�	E(K���A�$*'

learning_rate_1>J�:

loss_1֎>@�wϹ5       ��]�	�|>K���A�$*'

learning_rate_1>J�:

loss_1� I@��	5       ��]�	b�TK���A�$*'

learning_rate_1>J�:

loss_1��E@�z*r5       ��]�	ӺjK���A�$*'

learning_rate_1>J�:

loss_1�dE@�e�5       ��]�	�0�K���A�$*'

learning_rate_1>J�:

loss_1�BG@����5       ��]�	s��K���A�$*'

learning_rate_1>J�:

loss_1L;@V�\P5       ��]�	FX�K���A�$*'

learning_rate_1>J�:

loss_1ս2@[u��5       ��]�	��K���A�$*'

learning_rate_1>J�:

loss_1�,F@E�û5       ��]�	$��K���A�$*'

learning_rate_1>J�:

loss_1�9@O�5       ��]�	��K���A�$*'

learning_rate_1>J�:

loss_1�%G@��5       ��]�	%+L���A�$*'

learning_rate_1>J�:

loss_1��U@�a�5       ��]�	�/L���A�$*'

learning_rate_1>J�:

loss_1�A@�z
�5       ��]�	.�1L���A�$*'

learning_rate_1>J�:

loss_1�1L@�-»5       ��]�	\IL���A�$*'

learning_rate_1>J�:

loss_1~�-@�≹5       ��]�	�_L���A�$*'

learning_rate_1>J�:

loss_1��=@�{�5       ��]�	��uL���A�$*'

learning_rate_1>J�:

loss_1?C@)WT�5       ��]�	G�L���A�$*'

learning_rate_1>J�:

loss_1�E@&�w�5       ��]�	�V�L���A�$*'

learning_rate_1>J�:

loss_1i�F@���5       ��]�	��L���A�$*'

learning_rate_1>J�:

loss_1TbH@P���5       ��]�	&>�L���A�$*'

learning_rate_1>J�:

loss_1�*M@��D5       ��]�	ת�L���A�$*'

learning_rate_1>J�:

loss_1םD@�i~�5       ��]�	/��L���A�$*'

learning_rate_1>J�:

loss_1�B@'�s�5       ��]�	��M���A�$*'

learning_rate_1>J�:

loss_1�F@���5       ��]�	k(M���A�$*'

learning_rate_1>J�:

loss_1�hD@Z�5       ��]�	��>M���A�$*'

learning_rate_1>J�:

loss_1�!@dHH�5       ��]�	�sUM���A�$*'

learning_rate_1>J�:

loss_1-�:@t�j�5       ��]�	�kM���A�$*'

learning_rate_1>J�:

loss_1ZK@q�&�5       ��]�	��M���A�$*'

learning_rate_1>J�:

loss_1}=@��X5       ��]�	A�M���A�$*'

learning_rate_1>J�:

loss_1�]K@�h0V5       ��]�	+�M���A�$*'

learning_rate_1>J�:

loss_1̨6@M�Ġ5       ��]�	Nj�M���A�$*'

learning_rate_1>J�:

loss_1W�@@nq75       ��]�	�]�M���A�$*'

learning_rate_1>J�:

loss_1}�E@�.��5       ��]�	���M���A�$*'

learning_rate_1>J�:

loss_1��S@�� �5       ��]�	�		N���A�$*'

learning_rate_1>J�:

loss_1�J@خ�~5       ��]�	�`N���A�$*'

learning_rate_1>J�:

loss_1�_D@�%R�5       ��]�	T6N���A�$*'

learning_rate_1>J�:

loss_1�F@����5       ��]�	Y�LN���A�$*'

learning_rate_1>J�:

loss_1�D@@���z5       ��]�	�cN���A�$*'

learning_rate_1>J�:

loss_1�-P@�b`5       ��]�	�yN���A�$*'

learning_rate_1>J�:

loss_1��=@�}5       ��]�	O�N���A�$*'

learning_rate_1>J�:

loss_1C@���5       ��]�	Em�N���A�$*'

learning_rate_1>J�:

loss_1�N@�G^5       ��]�	i��N���A�$*'

learning_rate_1>J�:

loss_1��2@A\ͦ5       ��]�	(��N���A�$*'

learning_rate_1>J�:

loss_1w�L@g�5       ��]�	�N���A�$*'

learning_rate_1>J�:

loss_1��A@����5       ��]�	�S�N���A�$*'

learning_rate_1>J�:

loss_1�wA@=��<5       ��]�	��O���A�$*'

learning_rate_1>J�:

loss_1h[7@
G795       ��]�	�_,O���A�$*'

learning_rate_1>J�:

loss_1�f@@?��5       ��]�	��GO���A�$*'

learning_rate_1>J�:

loss_1Y�>@
S�f5       ��]�	'�]O���A�$*'

learning_rate_1>J�:

loss_1!~?@���x5       ��]�	� tO���A�$*'

learning_rate_1>J�:

loss_1d�=@g��"5       ��]�	�e�O���A�$*'

learning_rate_1>J�:

loss_1P�G@pO�5       ��]�	3�O���A�$*'

learning_rate_1>J�:

loss_1�0@v���5       ��]�	�ֵO���A�$*'

learning_rate_1>J�:

loss_1I�D@ߛ�5       ��]�	;8�O���A�$*'

learning_rate_1>J�:

loss_1=:@��>�5       ��]�	���O���A�$*'

learning_rate_1>J�:

loss_19}<@3�O�5       ��]�	h	�O���A�$*'

learning_rate_1>J�:

loss_1�!A@O �5       ��]�	��P���A�$*'

learning_rate_1>J�:

loss_1L�F@�O�E5       ��]�	F'P���A�$*'

learning_rate_1>J�:

loss_1:k?@x��o5       ��]�	�"=P���A�$*'

learning_rate_1>J�:

loss_1jLG@�ß5       ��]�	��SP���A�$*'

learning_rate_1>J�:

loss_1n�H@�h��5       ��]�	��iP���A�$*'

learning_rate_1>J�:

loss_1'A@^-5       ��]�	>��P���A�$*'

learning_rate_1>J�:

loss_10�F@����5       ��]�	H�P���A�$*'

learning_rate_1>J�:

loss_1F'Z@�h$D5       ��]�	�!�P���A�$*'

learning_rate_1>J�:

loss_1́I@��$x5       ��]�	}J�P���A�$*'

learning_rate_1>J�:

loss_1*�B@A��t5       ��]�	���P���A�$*'

learning_rate_1>J�:

loss_1D�U@�&O}5       ��]�	���P���A�$*'

learning_rate_1>J�:

loss_17�5@���5       ��]�	�}Q���A�$*'

learning_rate_1>J�:

loss_1�7@R\�5       ��]�	��Q���A�%*'

learning_rate_1>J�:

loss_1��E@mXS5       ��]�	��3Q���A�%*'

learning_rate_1>J�:

loss_1��F@"��5       ��]�	0JQ���A�%*'

learning_rate_1>J�:

loss_1��L@�I5       ��]�	��`Q���A�%*'

learning_rate_1>J�:

loss_1Q:@A��5       ��]�	hwQ���A�%*'

learning_rate_1>J�:

loss_1ՇB@����5       ��]�	�Q�Q���A�%*'

learning_rate_1>J�:

loss_1�O=@��P�5       ��]�	��Q���A�%*'

learning_rate_1>J�:

loss_1��?@���5       ��]�	��Q���A�%*'

learning_rate_1>J�:

loss_1߀'@ͩ�5       ��]�	L�Q���A�%*'

learning_rate_1>J�:

loss_1��8@��5       ��]�	]��Q���A�%*'

learning_rate_1>J�:

loss_1��H@�W�U5       ��]�	�U�Q���A�%*'

learning_rate_1>J�:

loss_1}�;@� �S5       ��]�	ǉR���A�%*'

learning_rate_1>J�:

loss_1vi@@#Yf5       ��]�	��*R���A�%*'

learning_rate_1>J�:

loss_1ǀE@�n�]5       ��]�	""AR���A�%*'

learning_rate_1>J�:

loss_1#I@5�v5       ��]�	ĔWR���A�%*'

learning_rate_1>J�:

loss_1 �5@	�a5       ��]�	��mR���A�%*'

learning_rate_1>J�:

loss_1��9@����5       ��]�	�T�R���A�%*'

learning_rate_1>J�:

loss_1�:G@�ԧ5       ��]�	���R���A�%*'

learning_rate_1>J�:

loss_1	6@�_5       ��]�	�ϱR���A�%*'

learning_rate_1>J�:

loss_1�E@8���5       ��]�	���R���A�%*'

learning_rate_1>J�:

loss_1��9@w�5       ��]�	���R���A�%*'

learning_rate_1>J�:

loss_1�17@�Q�5       ��]�	.��R���A�%*'

learning_rate_1>J�:

loss_1��H@�N5       ��]�	b
S���A�%*'

learning_rate_1>J�:

loss_1j<@<u5       ��]�	�; S���A�%*'

learning_rate_1>J�:

loss_1`�A@�Ƶ5       ��]�	e�6S���A�%*'

learning_rate_1>J�:

loss_1#!9@7W�t5       ��]�	,MS���A�%*'

learning_rate_1>J�:

loss_1�PB@��5       ��]�	Z^dS���A�%*'

learning_rate_1>J�:

loss_1�h8@�,�5       ��]�	�zS���A�%*'

learning_rate_1>J�:

loss_1�F9@9*��5       ��]�	��S���A�%*'

learning_rate_1>J�:

loss_1�29@
���5       ��]�	q�S���A�%*'

learning_rate_1>J�:

loss_10�H@��p5       ��]�	d=�S���A�%*'

learning_rate_1>J�:

loss_1�6@���R5       ��]�	j�S���A�%*'

learning_rate_1>J�:

loss_1��Q@�(L�5       ��]�	0d�S���A�%*'

learning_rate_1>J�:

loss_1	�/@�b�-5       ��]�	C$T���A�%*'

learning_rate_1>J�:

loss_1mZK@��Y5       ��]�	p_T���A�%*'

learning_rate_1>J�:

loss_1{�C@�!5       ��]�	ٳ.T���A�%*'

learning_rate_1>J�:

loss_16�B@�f �5       ��]�	�DT���A�%*'

learning_rate_1>J�:

loss_1�T@��_i5       ��]�	P[T���A�%*'

learning_rate_1>J�:

loss_1�=K@�P�5       ��]�	m�qT���A�%*'

learning_rate_1>J�:

loss_16�M@���5       ��]�	R$�T���A�%*'

learning_rate_1>J�:

loss_1�_:@t���5       ��]�	+�T���A�%*'

learning_rate_1>J�:

loss_1^�<@ݢ��5       ��]�	E2�T���A�%*'

learning_rate_1>J�:

loss_1x�3@���5       ��]�	C��T���A�%*'

learning_rate_1>J�:

loss_1x�J@���"5       ��]�	t��T���A�%*'

learning_rate_1>J�:

loss_1Zo@@�Q��5       ��]�	W�T���A�%*'

learning_rate_1>J�:

loss_1�k>@�Y�5       ��]�	�cU���A�%*'

learning_rate_1>J�:

loss_1	�C@�7[5       ��]�	g�%U���A�%*'

learning_rate_1>J�:

loss_1P�G@� 5       ��]�	�;U���A�%*'

learning_rate_1>J�:

loss_1�m?@�v�Z5       ��]�	��RU���A�%*'

learning_rate_1>J�:

loss_1�B@`�q5       ��]�	�jU���A�%*'

learning_rate_1>J�:

loss_1�<@]��5       ��]�	�D�U���A�%*'

learning_rate_1>J�:

loss_1!<7@!�;P5       ��]�	�v�U���A�%*'

learning_rate_1>J�:

loss_1�sR@[�!�5       ��]�	ͯ�U���A�%*'

learning_rate_1>J�:

loss_1�9@~؎5       ��]�	���U���A�%*'

learning_rate_1>J�:

loss_1#�<@uҿ5       ��]�	`#�U���A�%*'

learning_rate_1>J�:

loss_1ӫ6@�D��5       ��]�	6[�U���A�%*'

learning_rate_1>J�:

loss_1�bD@���5       ��]�	4�V���A�%*'

learning_rate_1>J�:

loss_1�2@ïm�5       ��]�	��V���A�%*'

learning_rate_1>J�:

loss_1�21@�XIs5       ��]�		�4V���A�%*'

learning_rate_1>J�:

loss_1|+6@�g�5       ��]�	��LV���A�%*'

learning_rate_1>J�:

loss_1�:,@T1'*5       ��]�	~cV���A�%*'

learning_rate_1>J�:

loss_1o`?@I�6 5       ��]�	�kyV���A�%*'

learning_rate_1>J�:

loss_1�I@���5       ��]�	��V���A�%*'

learning_rate_1>J�:

loss_1��B@�Z�45       ��]�	(b�V���A�%*'

learning_rate_1>J�:

loss_1�7B@��e)5       ��]�	��V���A�%*'

learning_rate_1>J�:

loss_1�a;@j�W5       ��]�	Y��V���A�%*'

learning_rate_1>J�:

loss_1��>@ ���5       ��]�	���V���A�%*'

learning_rate_1>J�:

loss_1S2@�%5       ��]�	�]W���A�%*'

learning_rate_1>J�:

loss_1�E@;U�5       ��]�	�W���A�%*'

learning_rate_1>J�:

loss_1��?@'2�5       ��]�	��-W���A�%*'

learning_rate_1>J�:

loss_10A@
;5       ��]�	01DW���A�%*'

learning_rate_1>J�:

loss_12XH@�S5       ��]�	9�ZW���A�%*'

learning_rate_1>J�:

loss_1��>@�/5       ��]�	�pW���A�%*'

learning_rate_1>J�:

loss_1_<=@h�Ӳ5       ��]�	v(�W���A�%*'

learning_rate_1>J�:

loss_1�s4@/�V55       ��]�	��W���A�%*'

learning_rate_1>J�:

loss_1�5F@t
\�5       ��]�	���W���A�%*'

learning_rate_1>J�:

loss_1�\<@# �#5       ��]�	��W���A�%*'

learning_rate_1>J�:

loss_1�mC@z�r�5       ��]�	��W���A�%*'

learning_rate_1>J�:

loss_1�w4@)΄�5       ��]�	��W���A�%*'

learning_rate_1>J�:

loss_15@� 5       ��]�	�cX���A�%*'

learning_rate_1>J�:

loss_1\�I@�n�5       ��]�	��&X���A�%*'

learning_rate_1>J�:

loss_1�F@{'��5       ��]�	�N>X���A�%*'

learning_rate_1>J�:

loss_1�uG@g3��5       ��]�	{TX���A�%*'

learning_rate_1>J�:

loss_1P�P@����5       ��]�	z�jX���A�%*'

learning_rate_1>J�:

loss_1hr8@��"5       ��]�	�ˀX���A�%*'

learning_rate_1>J�:

loss_1;�F@�fq�5       ��]�	@�X���A�%*'

learning_rate_1>J�:

loss_1��:@	��5       ��]�	;u�X���A�%*'

learning_rate_1>J�:

loss_1�h6@�i�5       ��]�	4f�X���A�%*'

learning_rate_1>J�:

loss_1a�?@^DUN5       ��]�	6��X���A�%*'

learning_rate_1>J�:

loss_1XI@·�j5       ��]�	��X���A�%*'

learning_rate_1>J�:

loss_1J3@H�چ5       ��]�	�UY���A�%*'

learning_rate_1>J�:

loss_1w�?@�=��5       ��]�	6�Y���A�%*'

learning_rate_1>J�:

loss_1�mH@a�(t5       ��]�	��4Y���A�%*'

learning_rate_1>J�:

loss_1��8@n+�5       ��]�	xKY���A�%*'

learning_rate_1>J�:

loss_1��:@�X��5       ��]�	E)aY���A�%*'

learning_rate_1>J�:

loss_1<A?@a�5       ��]�	 QwY���A�%*'

learning_rate_1>J�:

loss_1��@@1xF5       ��]�	���Y���A�%*'

learning_rate_1>J�:

loss_1��=@�i��5       ��]�	�#�Y���A�%*'

learning_rate_1>J�:

loss_1�:@�Q�5       ��]�	��Y���A�%*'

learning_rate_1>J�:

loss_1��A@EQ�5       ��]�	�]�Y���A�%*'

learning_rate_1>J�:

loss_1h�C@��5       ��]�	��Y���A�%*'

learning_rate_1>J�:

loss_10I@�ҥ�5       ��]�	'a�Y���A�%*'

learning_rate_1>J�:

loss_1Kl?@S�CS5       ��]�	L�Z���A�%*'

learning_rate_1>J�:

loss_1(�3@FyB�5       ��]�	-6-Z���A�%*'

learning_rate_1>J�:

loss_1��E@���5       ��]�	�TDZ���A�%*'

learning_rate_1>J�:

loss_1�|L@^�)5       ��]�	v-[Z���A�%*'

learning_rate_1>J�:

loss_1�K@װ�5       ��]�	Y�qZ���A�%*'

learning_rate_1>J�:

loss_1;L@�a�5       ��]�	�w�Z���A�%*'

learning_rate_1>J�:

loss_1|uI@mי>5       ��]�	|v�Z���A�%*'

learning_rate_1>J�:

loss_1^|?@?{�5       ��]�	��Z���A�%*'

learning_rate_1>J�:

loss_1�A@}��5       ��]�	5�Z���A�%*'

learning_rate_1>J�:

loss_1YfI@���U5       ��]�	�L�Z���A�%*'

learning_rate_1>J�:

loss_1+A@d?R�5       ��]�	̏�Z���A�%*'

learning_rate_1>J�:

loss_1�$C@�I��5       ��]�	@)[���A�%*'

learning_rate_1>J�:

loss_1��7@E���5       ��]�	��#[���A�%*'

learning_rate_1>J�:

loss_1�#<@�@ˉ5       ��]�	*:[���A�%*'

learning_rate_1>J�:

loss_1��;@�)�g5       ��]�	��P[���A�%*'

learning_rate_1>J�:

loss_1�!C@�W�(5       ��]�	f�f[���A�%*'

learning_rate_1>J�:

loss_1��A@;r��5       ��]�	�}[���A�%*'

learning_rate_1>J�:

loss_1]�;@��5       ��]�	ѓ[���A�%*'

learning_rate_1>J�:

loss_1 i5@r*a15       ��]�	y�[���A�%*'

learning_rate_1>J�:

loss_19�;@��O�5       ��]�	���[���A�%*'

learning_rate_1>J�:

loss_1�;@�7��5       ��]�	��[���A�%*'

learning_rate_1>J�:

loss_1*bL@f;��5       ��]�	s�[���A�%*'

learning_rate_1>J�:

loss_1�AF@�/��5       ��]�	�
\���A�%*'

learning_rate_1>J�:

loss_1�uD@x��5       ��]�	�5\���A�%*'

learning_rate_1>J�:

loss_1�]<@���5       ��]�	�v1\���A�%*'

learning_rate_1>J�:

loss_1�:@��rp5       ��]�	�;H\���A�%*'

learning_rate_1>J�:

loss_1f�@@�� �5       ��]�	��^\���A�&*'

learning_rate_1>J�:

loss_16hE@�R�u5       ��]�	�t\���A�&*'

learning_rate_1>J�:

loss_1�7K@�\#5       ��]�	�\���A�&*'

learning_rate_1>J�:

loss_1Ȟ:@�u��5       ��]�	�b�\���A�&*'

learning_rate_1>J�:

loss_1�9D@Et�S5       ��]�	̟�\���A�&*'

learning_rate_1>J�:

loss_1�yA@�C�5       ��]�	��\���A�&*'

learning_rate_1>J�:

loss_1�b@@���5       ��]�	D��\���A�&*'

learning_rate_1>J�:

loss_1��E@>�*5       ��]�	���\���A�&*'

learning_rate_1>J�:

loss_1X"J@s&��5       ��]�		X]���A�&*'

learning_rate_1>J�:

loss_1�:@�5       ��]�	f�']���A�&*'

learning_rate_1>J�:

loss_1�(@Atw]5       ��]�		?]���A�&*'

learning_rate_1>J�:

loss_1�@J@�d�5       ��]�	�2U]���A�&*'

learning_rate_1>J�:

loss_1i{:@x���5       ��]�	��k]���A�&*'

learning_rate_1>J�:

loss_1PJ@�d�5       ��]�	㘁]���A�&*'

learning_rate_1>J�:

loss_1�sA@��*5       ��]�	���]���A�&*'

learning_rate_1>J�:

loss_1]i)@G�h5       ��]�	A�]���A�&*'

learning_rate_1>J�:

loss_1��@@��ŀ5       ��]�	�0�]���A�&*'

learning_rate_1>J�:

loss_1�SJ@�`wo5       ��]�	bC�]���A�&*'

learning_rate_1>J�:

loss_1�)4@�!5       ��]�	�]���A�&*'

learning_rate_1>J�:

loss_1��F@�+��5       ��]�	N�^���A�&*'

learning_rate_1>J�:

loss_1��E@�O�Z5       ��]�	;&^���A�&*'

learning_rate_1>J�:

loss_1�79@�Jq5       ��]�	�C4^���A�&*'

learning_rate_1>J�:

loss_1�9@�<L5       ��]�	��J^���A�&*'

learning_rate_1>J�:

loss_1�5H@z$5       ��]�	}�`^���A�&*'

learning_rate_1>J�:

loss_1+G@~5�5       ��]�	�(w^���A�&*'

learning_rate_1>J�:

loss_1�E@���5       ��]�	P�^���A�&*'

learning_rate_1>J�:

loss_19g?@C�Dm5       ��]�	��^���A�&*'

learning_rate_1>J�:

loss_1N@?��K5       ��]�	z��^���A�&*'

learning_rate_1>J�:

loss_1��J@,��5       ��]�	m6�^���A�&*'

learning_rate_1>J�:

loss_1P�7@�f�j5       ��]�	vF�^���A�&*'

learning_rate_1>J�:

loss_1�E@��M�5       ��]�	,��^���A�&*'

learning_rate_1>J�:

loss_1tY:@�f�^5       ��]�	R�_���A�&*'

learning_rate_1>J�:

loss_1�U;@�3M5       ��]�	�l)_���A�&*'

learning_rate_1>J�:

loss_1�;@LQ�05       ��]�	�R?_���A�&*'

learning_rate_1>J�:

loss_1ĻR@0���5       ��]�	��U_���A�&*'

learning_rate_1>J�:

loss_1�xI@����5       ��]�	�l_���A�&*'

learning_rate_1>J�:

loss_1a�=@���5       ��]�	n�_���A�&*'

learning_rate_1>J�:

loss_1C�L@��5�5       ��]�	�_���A�&*'

learning_rate_1>J�:

loss_12]6@<�}�5       ��]�	۟�_���A�&*'

learning_rate_1>J�:

loss_1�C@x�j5       ��]�	X��_���A�&*'

learning_rate_1>J�:

loss_1Q<@tB�95       ��]�	X��_���A�&*'

learning_rate_1>J�:

loss_1E�@@��>5       ��]�	2��_���A�&*'

learning_rate_1>J�:

loss_1%�3@205       ��]�	��`���A�&*'

learning_rate_1>J�:

loss_1	�2@��y�5       ��]�	&"`���A�&*'

learning_rate_1>J�:

loss_1�uA@E;��5       ��]�	Ղ8`���A�&*'

learning_rate_1>J�:

loss_1��A@Ĉ�~5       ��]�	@�N`���A�&*'

learning_rate_1>J�:

loss_1�A@آ>-5       ��]�	CPe`���A�&*'

learning_rate_1>J�:

loss_1��<@�4RD5       ��]�	�6{`���A�&*'

learning_rate_1>J�:

loss_1i�B@�D��5       ��]�	j��`���A�&*'

learning_rate_1>J�:

loss_1�5;@V''�5       ��]�	��`���A�&*'

learning_rate_1>J�:

loss_1
%E@��v5       ��]�	���`���A�&*'

learning_rate_1>J�:

loss_1}�B@x��N5       ��]�	L��`���A�&*'

learning_rate_1>J�:

loss_1��A@xI�_5       ��]�	���`���A�&*'

learning_rate_1>J�:

loss_1s�8@�Pˢ5       ��]�	d� a���A�&*'

learning_rate_1>J�:

loss_1�F@�Ks�5       ��]�	{#a���A�&*'

learning_rate_1>J�:

loss_1��;@�p��5       ��]�	�-a���A�&*'

learning_rate_1>J�:

loss_1vC@1%e�5       ��]�	y�Ca���A�&*'

learning_rate_1>J�:

loss_1E�J@)���5       ��]�	��Za���A�&*'

learning_rate_1>J�:

loss_1��5@.��B5       ��]�	�Tqa���A�&*'

learning_rate_1>J�:

loss_1�9@�E�5       ��]�	���a���A�&*'

learning_rate_1>J�:

loss_1�E<@(���5       ��]�	�ǝa���A�&*'

learning_rate_1>J�:

loss_1Jh8@�ޘ5       ��]�	^�a���A�&*'

learning_rate_1>J�:

loss_1h�A@�M._5       ��]�	���a���A�&*'

learning_rate_1>J�:

loss_1.�-@��E5       ��]�	���a���A�&*'

learning_rate_1>J�:

loss_1�\N@���5       ��]�	p��a���A�&*'

learning_rate_1>J�:

loss_1��=@�h��5       ��]�	�%b���A�&*'

learning_rate_1>J�:

loss_1#�8@CIj�5       ��]�	�>$b���A�&*'

learning_rate_1>J�:

loss_1%�E@�?�5       ��]�	�k:b���A�&*'

learning_rate_1>J�:

loss_1�$M@�S5       ��]�	��Pb���A�&*'

learning_rate_1>J�:

loss_1��E@�O(5       ��]�	gb���A�&*'

learning_rate_1>J�:

loss_1��L@Jq�`5       ��]�	,�}b���A�&*'

learning_rate_1>J�:

loss_1�F=@�-�=5       ��]�	wJ�b���A�&*'

learning_rate_1>J�:

loss_1��?@|.La5       ��]�	Wx�b���A�&*'

learning_rate_1>J�:

loss_1��3@G��,5       ��]�	���b���A�&*'

learning_rate_1>J�:

loss_1q�C@|F�A5       ��]�	�L�b���A�&*'

learning_rate_1>J�:

loss_16D@�ג25       ��]�	o�b���A�&*'

learning_rate_1>J�:

loss_1
1=@u�/5       ��]�	�c���A�&*'

learning_rate_1>J�:

loss_1�fB@���5       ��]�	Rdc���A�&*'

learning_rate_1>J�:

loss_1P%E@�(�5       ��]�	�1c���A�&*'

learning_rate_1>J�:

loss_1��>@a6�5       ��]�	I�Gc���A�&*'

learning_rate_1>J�:

loss_1k~4@%eV�5       ��]�	$$^c���A�&*'

learning_rate_1>J�:

loss_1�};@�0z5       ��]�	�2tc���A�&*'

learning_rate_1>J�:

loss_1|�;@C��5       ��]�	���c���A�&*'

learning_rate_1>J�:

loss_1�I:@T�b�5       ��]�	E͡c���A�&*'

learning_rate_1>J�:

loss_1M�O@�}�5       ��]�	 ��c���A�&*'

learning_rate_1>J�:

loss_1�hA@G�So5       ��]�	�c���A�&*'

learning_rate_1>J�:

loss_1��M@���5       ��]�	�B�c���A�&*'

learning_rate_1>J�:

loss_1`8@]D��5       ��]�	ɍ�c���A�&*'

learning_rate_1>J�:

loss_1&A@5J)�5       ��]�	��d���A�&*'

learning_rate_1>J�:

loss_1/�I@y,j5       ��]�	�v)d���A�&*'

learning_rate_1>J�:

loss_1��;@{w'5       ��]�	߮?d���A�&*'

learning_rate_1>J�:

loss_1��7@�2.�5       ��]�	aVd���A�&*'

learning_rate_1>J�:

loss_1ǿ<@�_�5       ��]�	�ld���A�&*'

learning_rate_1>J�:

loss_1 �F@�ħ5       ��]�	�<�d���A�&*'

learning_rate_1>J�:

loss_1��5@o�[�5       ��]�	�k�d���A�&*'

learning_rate_1>J�:

loss_1eXF@!��5       ��]�	H��d���A�&*'

learning_rate_1>J�:

loss_1�Q@�0�5       ��]�	&�d���A�&*'

learning_rate_1>J�:

loss_1�89@Ą�:5       ��]�	J2�d���A�&*'

learning_rate_1>J�:

loss_1�<@�$A|5       ��]�	Z�d���A�&*'

learning_rate_1>J�:

loss_1�EJ@���5       ��]�	ڟe���A�&*'

learning_rate_1>J�:

loss_1P;B@Ye�r5       ��]�	y�e���A�&*'

learning_rate_1>J�:

loss_1r�B@qU�5       ��]�	�4e���A�&*'

learning_rate_1>J�:

loss_1�XG@Z �5       ��]�	KIJe���A�&*'

learning_rate_1>J�:

loss_1�M@�?�H5       ��]�	ks`e���A�&*'

learning_rate_1>J�:

loss_1�+V@oT	5       ��]�	�ve���A�&*'

learning_rate_1>J�:

loss_17�K@F���5       ��]�	F�e���A�&*'

learning_rate_1>J�:

loss_1ÈM@⋵5       ��]�	Ј�e���A�&*'

learning_rate_1>J�:

loss_1��8@	m�-5       ��]�	�r�e���A�&*'

learning_rate_1>J�:

loss_1nO@@-��5       ��]�	֞�e���A�&*'

learning_rate_1>J�:

loss_1�7C@5���5       ��]�	R��e���A�&*'

learning_rate_1>J�:

loss_1��2@h���5       ��]�	;�e���A�&*'

learning_rate_1>J�:

loss_1��6@�D!�5       ��]�	?]f���A�&*'

learning_rate_1>J�:

loss_1DC@�|�5       ��]�	~�+f���A�&*'

learning_rate_1>J�:

loss_1��9@�3�5       ��]�	~Bf���A�&*'

learning_rate_1>J�:

loss_1�-9@��~5       ��]�	a�Xf���A�&*'

learning_rate_1>J�:

loss_1��:@Z�?5       ��]�	(�nf���A�&*'

learning_rate_1>J�:

loss_1+ED@�=b5       ��]�	�3�f���A�&*'

learning_rate_1>J�:

loss_1�4@�kE5       ��]�	rj�f���A�&*'

learning_rate_1>J�:

loss_1��@@ا�5       ��]�	`�f���A�&*'

learning_rate_1>J�:

loss_1��B@�5�|5       ��]�	�f���A�&*'

learning_rate_1>J�:

loss_1b8@:�Ζ5       ��]�	'H�f���A�&*'

learning_rate_1>J�:

loss_1h�C@�_5       ��]�	|x�f���A�&*'

learning_rate_1>J�:

loss_14N@Q�*[5       ��]�	�
g���A�&*'

learning_rate_1>J�:

loss_1�>@����5       ��]�	NX!g���A�&*'

learning_rate_1>J�:

loss_1�F@��5       ��]�	��7g���A�&*'

learning_rate_1>J�:

loss_1e8M@I�EY5       ��]�	2�Mg���A�&*'

learning_rate_1>J�:

loss_1;9@�=I�5       ��]�	�eg���A�&*'

learning_rate_1>J�:

loss_1LE@� �5       ��]�	��{g���A�&*'

learning_rate_1>J�:

loss_1}�?@�7��5       ��]�	(�g���A�'*'

learning_rate_1>J�:

loss_1p@@��[5       ��]�	��g���A�'*'

learning_rate_1>J�:

loss_1�R3@��(�5       ��]�	�w�g���A�'*'

learning_rate_1>J�:

loss_1�8:@��,�5       ��]�	���g���A�'*'

learning_rate_1>J�:

loss_1)@P@V�G�5       ��]�	�Z�g���A�'*'

learning_rate_1>J�:

loss_19<0@�q��5       ��]�	H�h���A�'*'

learning_rate_1>J�:

loss_1�/L@<Th5       ��]�	aih���A�'*'

learning_rate_1>J�:

loss_1~�E@�2�z5       ��]�	Ԁ/h���A�'*'

learning_rate_1>J�:

loss_1�dH@�a�5       ��]�	�#Fh���A�'*'

learning_rate_1>J�:

loss_1oI@@���5       ��]�	�]h���A�'*'

learning_rate_1>J�:

loss_132@�"��5       ��]�	Csh���A�'*'

learning_rate_1>J�:

loss_1΁C@��5       ��]�	���h���A�'*'

learning_rate_1>J�:

loss_1�=;@�@ a5       ��]�	���h���A�'*'

learning_rate_1>J�:

loss_1pN=@q5       ��]�	�?�h���A�'*'

learning_rate_1>J�:

loss_1J�-@��L�5       ��]�	B��h���A�'*'

learning_rate_1>J�:

loss_1%+8@	9l�5       ��]�	H[�h���A�'*'

learning_rate_1>J�:

loss_1��A@#75       ��]�	%��h���A�'*'

learning_rate_1>J�:

loss_1�sZ@��G�5       ��]�	��i���A�'*'

learning_rate_1>J�:

loss_1
�5@(�5       ��]�	�(i���A�'*'

learning_rate_1>J�:

loss_1�;@vF5       ��]�	{�>i���A�'*'

learning_rate_1>J�:

loss_1�<@0�x5       ��]�	��Ui���A�'*'

learning_rate_1>J�:

loss_127@� 
5       ��]�	Sli���A�'*'

learning_rate_1>J�:

loss_1EW=@u��t5       ��]�	�i���A�'*'

learning_rate_1>J�:

loss_1��;@��5       ��]�	唚i���A�'*'

learning_rate_1>J�:

loss_1�5A@�<�E5       ��]�	��i���A�'*'

learning_rate_1>J�:

loss_1_�E@|��5       ��]�	/]�i���A�'*'

learning_rate_1>J�:

loss_1l�B@伖�5       ��]�	'
�i���A�'*'

learning_rate_1>J�:

loss_1�2>@�G��5       ��]�	��i���A�'*'

learning_rate_1>J�:

loss_1�%H@҇a�5       ��]�	��j���A�'*'

learning_rate_1>J�:

loss_1o�G@�K�5       ��]�	�"j���A�'*'

learning_rate_1>J�:

loss_1~cF@�9��5       ��]�	��8j���A�'*'

learning_rate_1>J�:

loss_1��C@�n5       ��]�	�-Oj���A�'*'

learning_rate_1>J�:

loss_17@=@��5       ��]�	xoej���A�'*'

learning_rate_1>J�:

loss_1&?@gzlQ5       ��]�	�|j���A�'*'

learning_rate_1>J�:

loss_1�7@�<�%5       ��]�	�@�j���A�'*'

learning_rate_1>J�:

loss_1�:<@��5       ��]�	
��j���A�'*'

learning_rate_1>J�:

loss_1+mH@�?{5       ��]�	�߾j���A�'*'

learning_rate_1>J�:

loss_1�p<@��8�5       ��]�	��j���A�'*'

learning_rate_1>J�:

loss_1�^D@
F�5       ��]�	g
�j���A�'*'

learning_rate_1>J�:

loss_1}/@��W5       ��]�	'�k���A�'*'

learning_rate_1>J�:

loss_14�>@���5       ��]�	-k���A�'*'

learning_rate_1>J�:

loss_1=�6@�Z�`5       ��]�	AD/k���A�'*'

learning_rate_1>J�:

loss_1��G@��5       ��]�	s�Ek���A�'*'

learning_rate_1>J�:

loss_1�G@uP�v5       ��]�	\k���A�'*'

learning_rate_1>J�:

loss_1ûK@�A��5       ��]�	?Hrk���A�'*'

learning_rate_1>J�:

loss_1��6@@f�75       ��]�	���k���A�'*'

learning_rate_1>J�:

loss_1��E@��$5       ��]�	$�k���A�'*'

learning_rate_1>J�:

loss_1WS2@�� �5       ��]�	>y�k���A�'*'

learning_rate_1>J�:

loss_1ڃ=@����5       ��]�	nw�k���A�'*'

learning_rate_1>J�:

loss_1� 4@�ֻ