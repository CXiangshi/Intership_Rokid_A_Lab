       ŁK"	  :ŢĘÖAbrain.Event:2Z$ŇMd     <ö}j	XÓ:ŢĘÖA"ŔČ	
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:@˙˙˙˙˙˙˙˙˙*
shape:@˙˙˙˙˙˙˙˙˙
v
input/Placeholder_1Placeholder*
dtype0*'
_output_shapes
:@˙˙˙˙˙˙˙˙˙*
shape:@˙˙˙˙˙˙˙˙˙
i
$learning_rate/Variable/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
z
learning_rate/Variable
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ú
learning_rate/Variable/AssignAssignlearning_rate/Variable$learning_rate/Variable/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*)
_class
loc:@learning_rate/Variable

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

;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:@*
dtype0*
_output_shapes
:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Á
<rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concatConcatV2;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_1Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zerosFill<rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concatArnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes
:	@

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:@*
dtype0*
_output_shapes
:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:@*
dtype0*
_output_shapes
:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ç
>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1ConcatV2=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_4=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_5Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1Fill>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	@

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:@

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ConstConst*
valueB:@*
dtype0*
_output_shapes
:

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
É
>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatConcatV2=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_1Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0

Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zerosFill>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatCrnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/Const*
T0*

index_type0*
_output_shapes
:	@

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_2Const*
dtype0*
_output_shapes
:*
valueB:@

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_3Const*
dtype0*
_output_shapes
:*
valueB:

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_4Const*
valueB:@*
dtype0*
_output_shapes
:

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ď
@rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1ConcatV2?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_4?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_5Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0

Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1Fill@rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	@

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_6Const*
valueB:@*
dtype0*
_output_shapes
:

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

*embedding/Initializer/random_uniform/shapeConst*
_class
loc:@embedding*
valueB"Ţ     *
dtype0*
_output_shapes
:

(embedding/Initializer/random_uniform/minConst*
_class
loc:@embedding*
valueB
 *Yţź*
dtype0*
_output_shapes
: 

(embedding/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *Yţ<
Ţ
2embedding/Initializer/random_uniform/RandomUniformRandomUniform*embedding/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
Ţ/*

seed *
T0*
_class
loc:@embedding*
seed2 
Â
(embedding/Initializer/random_uniform/subSub(embedding/Initializer/random_uniform/max(embedding/Initializer/random_uniform/min*
T0*
_class
loc:@embedding*
_output_shapes
: 
Ö
(embedding/Initializer/random_uniform/mulMul2embedding/Initializer/random_uniform/RandomUniform(embedding/Initializer/random_uniform/sub*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Č
$embedding/Initializer/random_uniformAdd(embedding/Initializer/random_uniform/mul(embedding/Initializer/random_uniform/min*
_class
loc:@embedding* 
_output_shapes
:
Ţ/*
T0

	embedding
VariableV2* 
_output_shapes
:
Ţ/*
shared_name *
_class
loc:@embedding*
	container *
shape:
Ţ/*
dtype0
˝
embedding/AssignAssign	embedding$embedding/Initializer/random_uniform*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/*
use_locking(
n
embedding/readIdentity	embedding*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
x
lm/embedding_lookup/axisConst*
_class
loc:@embedding*
value	B : *
dtype0*
_output_shapes
: 
Ô
lm/embedding_lookupGatherV2embedding/readinput/Placeholderlm/embedding_lookup/axis*
Taxis0*
Tindices0*
Tparams0*
_class
loc:@embedding*,
_output_shapes
:@˙˙˙˙˙˙˙˙˙
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
lm/rnn/rangeRangelm/rnn/range/startlm/rnn/Ranklm/rnn/range/delta*

Tidx0*
_output_shapes
:
g
lm/rnn/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
T
lm/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

lm/rnn/concatConcatV2lm/rnn/concat/values_0lm/rnn/rangelm/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

lm/rnn/transpose	Transposelm/embedding_lookuplm/rnn/concat*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
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

lm/rnn/strided_sliceStridedSlicelm/rnn/Shapelm/rnn/strided_slice/stacklm/rnn/strided_slice/stack_1lm/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
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
valueB:
V
lm/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

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
lm/rnn/zerosFilllm/rnn/concat_1lm/rnn/zeros/Const*
_output_shapes
:	@*
T0*

index_type0
M
lm/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 

lm/rnn/TensorArrayTensorArrayV3lm/rnn/strided_slice*
element_shape:	@*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*2
tensor_array_namelm/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 

lm/rnn/TensorArray_1TensorArrayV3lm/rnn/strided_slice*1
tensor_array_namelm/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:	@*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
o
lm/rnn/TensorArrayUnstack/ShapeShapelm/rnn/transpose*
T0*
out_type0*
_output_shapes
:
w
-lm/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/lm/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
y
/lm/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ű
'lm/rnn/TensorArrayUnstack/strided_sliceStridedSlicelm/rnn/TensorArrayUnstack/Shape-lm/rnn/TensorArrayUnstack/strided_slice/stack/lm/rnn/TensorArrayUnstack/strided_slice/stack_1/lm/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
g
%lm/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
g
%lm/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Đ
lm/rnn/TensorArrayUnstack/rangeRange%lm/rnn/TensorArrayUnstack/range/start'lm/rnn/TensorArrayUnstack/strided_slice%lm/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

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
lm/rnn/MaximumMaximumlm/rnn/Maximum/xlm/rnn/strided_slice*
T0*
_output_shapes
: 
`
lm/rnn/MinimumMinimumlm/rnn/strided_slicelm/rnn/Maximum*
_output_shapes
: *
T0
`
lm/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
ś
lm/rnn/while/EnterEnterlm/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
Ľ
lm/rnn/while/Enter_1Enterlm/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
Ž
lm/rnn/while/Enter_2Enterlm/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
Ţ
lm/rnn/while/Enter_3Enter;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes
:	@**

frame_namelm/rnn/while/while_context*
T0*
is_constant( 
ŕ
lm/rnn/while/Enter_4Enter=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@**

frame_namelm/rnn/while/while_context
ŕ
lm/rnn/while/Enter_5Enter=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros*
is_constant( *
parallel_iterations *
_output_shapes
:	@**

frame_namelm/rnn/while/while_context*
T0
â
lm/rnn/while/Enter_6Enter?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@**

frame_namelm/rnn/while/while_context
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
lm/rnn/while/Merge_2Mergelm/rnn/while/Enter_2lm/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

lm/rnn/while/Merge_3Mergelm/rnn/while/Enter_3lm/rnn/while/NextIteration_3*
T0*
N*!
_output_shapes
:	@: 

lm/rnn/while/Merge_4Mergelm/rnn/while/Enter_4lm/rnn/while/NextIteration_4*
N*!
_output_shapes
:	@: *
T0

lm/rnn/while/Merge_5Mergelm/rnn/while/Enter_5lm/rnn/while/NextIteration_5*
T0*
N*!
_output_shapes
:	@: 

lm/rnn/while/Merge_6Mergelm/rnn/while/Enter_6lm/rnn/while/NextIteration_6*
T0*
N*!
_output_shapes
:	@: 
g
lm/rnn/while/LessLesslm/rnn/while/Mergelm/rnn/while/Less/Enter*
_output_shapes
: *
T0
ą
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
­
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

lm/rnn/while/SwitchSwitchlm/rnn/while/Mergelm/rnn/while/LoopCond*
T0*%
_class
loc:@lm/rnn/while/Merge*
_output_shapes
: : 

lm/rnn/while/Switch_1Switchlm/rnn/while/Merge_1lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_1*
_output_shapes
: : 

lm/rnn/while/Switch_2Switchlm/rnn/while/Merge_2lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_2*
_output_shapes
: : 
Ş
lm/rnn/while/Switch_3Switchlm/rnn/while/Merge_3lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_3**
_output_shapes
:	@:	@
Ş
lm/rnn/while/Switch_4Switchlm/rnn/while/Merge_4lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_4**
_output_shapes
:	@:	@
Ş
lm/rnn/while/Switch_5Switchlm/rnn/while/Merge_5lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_5**
_output_shapes
:	@:	@
Ş
lm/rnn/while/Switch_6Switchlm/rnn/while/Merge_6lm/rnn/while/LoopCond**
_output_shapes
:	@:	@*
T0*'
_class
loc:@lm/rnn/while/Merge_6
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
lm/rnn/while/Identity_2Identitylm/rnn/while/Switch_2:1*
_output_shapes
: *
T0
f
lm/rnn/while/Identity_3Identitylm/rnn/while/Switch_3:1*
_output_shapes
:	@*
T0
f
lm/rnn/while/Identity_4Identitylm/rnn/while/Switch_4:1*
T0*
_output_shapes
:	@
f
lm/rnn/while/Identity_5Identitylm/rnn/while/Switch_5:1*
T0*
_output_shapes
:	@
f
lm/rnn/while/Identity_6Identitylm/rnn/while/Switch_6:1*
_output_shapes
:	@*
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
Č
lm/rnn/while/TensorArrayReadV3TensorArrayReadV3$lm/rnn/while/TensorArrayReadV3/Enterlm/rnn/while/Identity_1&lm/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	@
Â
$lm/rnn/while/TensorArrayReadV3/EnterEnterlm/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
í
&lm/rnn/while/TensorArrayReadV3/Enter_1EnterAlm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
ç
Qrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *óľ˝*
dtype0*
_output_shapes
: 
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *óľ=*
dtype0*
_output_shapes
: 
Ó
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
Ţ
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: 
ň
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ä
Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:

í
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ů
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:


5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
*
T0
Ň
@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zerosConst*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB*    *
dtype0
ß
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ă
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:

3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
_output_shapes	
:

<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/ConstConst^lm/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :

Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axisConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concatConcatV2lm/rnn/while/TensorArrayReadV3lm/rnn/while/Identity_4Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axis*
N*
_output_shapes
:	@*

Tidx0*
T0

=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMulMatMul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concatClm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter*
_output_shapes
:	@*
transpose_a( *
transpose_b( *
T0

Clm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations * 
_output_shapes
:
**

frame_namelm/rnn/while/while_context*
T0*
is_constant(

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAddBiasAdd=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMulDlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	@

Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
is_constant(*
parallel_iterations *
_output_shapes	
:**

frame_namelm/rnn/while/while_context*
T0

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_1Const^lm/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0

<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/splitSplit<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd*
T0*@
_output_shapes.
,:	@:	@:	@:	@*
	num_split

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2Const^lm/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ë
:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/AddAdd>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:2>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	@
Ż
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/SigmoidSigmoid:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add*
_output_shapes
:	@*
T0
Ä
:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MulMullm/rnn/while/Identity_3>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes
:	@*
T0
ł
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split*
T0*
_output_shapes
:	@
­
;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/TanhTanh>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:1*
T0*
_output_shapes
:	@
ě
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1Mul@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	@
ç
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1Add:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1*
_output_shapes
:	@*
T0
­
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1Tanh<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	@
ľ
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2Sigmoid>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:3*
T0*
_output_shapes
:	@
î
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2Mul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*
_output_shapes
:	@*
T0

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axisConst^lm/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
§
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1ConcatV2<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2lm/rnn/while/Identity_6Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axis*
N*
_output_shapes
:	@*

Tidx0*
T0

?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1MatMul?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1Clm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes
:	@*
transpose_a( *
transpose_b( 

@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1BiasAdd?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	@*
T0

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_4Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ľ
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1Split>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1*
T0*@
_output_shapes.
,:	@:	@:	@:	@*
	num_split

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5Const^lm/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ď
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2Add@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:2>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5*
T0*
_output_shapes
:	@
ł
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3Sigmoid<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2*
T0*
_output_shapes
:	@
Č
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3Mullm/rnn/while/Identity_5@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*
T0*
_output_shapes
:	@
ľ
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4Sigmoid>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1*
T0*
_output_shapes
:	@
ą
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2Tanh@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:1*
T0*
_output_shapes
:	@
î
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4Mul@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*
T0*
_output_shapes
:	@
é
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3Add<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4*
_output_shapes
:	@*
T0
­
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3Tanh<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3*
T0*
_output_shapes
:	@
ˇ
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5Sigmoid@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:3*
_output_shapes
:	@*
T0
î
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5Mul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*
T0*
_output_shapes
:	@
ŕ
0lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterlm/rnn/while/Identity_1<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5lm/rnn/while/Identity_2*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
_output_shapes
: 
Ł
6lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlm/rnn/TensorArray*
is_constant(*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
parallel_iterations 
n
lm/rnn/while/add_1/yConst^lm/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
i
lm/rnn/while/add_1Addlm/rnn/while/Identity_1lm/rnn/while/add_1/y*
_output_shapes
: *
T0
^
lm/rnn/while/NextIterationNextIterationlm/rnn/while/add*
T0*
_output_shapes
: 
b
lm/rnn/while/NextIteration_1NextIterationlm/rnn/while/add_1*
T0*
_output_shapes
: 

lm/rnn/while/NextIteration_2NextIteration0lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

lm/rnn/while/NextIteration_3NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1*
_output_shapes
:	@*
T0

lm/rnn/while/NextIteration_4NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2*
T0*
_output_shapes
:	@

lm/rnn/while/NextIteration_5NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3*
_output_shapes
:	@*
T0

lm/rnn/while/NextIteration_6NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
T0*
_output_shapes
:	@
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
lm/rnn/while/Exit_2Exitlm/rnn/while/Switch_2*
_output_shapes
: *
T0
\
lm/rnn/while/Exit_3Exitlm/rnn/while/Switch_3*
_output_shapes
:	@*
T0
\
lm/rnn/while/Exit_4Exitlm/rnn/while/Switch_4*
T0*
_output_shapes
:	@
\
lm/rnn/while/Exit_5Exitlm/rnn/while/Switch_5*
T0*
_output_shapes
:	@
\
lm/rnn/while/Exit_6Exitlm/rnn/while/Switch_6*
T0*
_output_shapes
:	@
Ś
)lm/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lm/rnn/TensorArraylm/rnn/while/Exit_2*
_output_shapes
: *%
_class
loc:@lm/rnn/TensorArray

#lm/rnn/TensorArrayStack/range/startConst*%
_class
loc:@lm/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

#lm/rnn/TensorArrayStack/range/deltaConst*%
_class
loc:@lm/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
ó
lm/rnn/TensorArrayStack/rangeRange#lm/rnn/TensorArrayStack/range/start)lm/rnn/TensorArrayStack/TensorArraySizeV3#lm/rnn/TensorArrayStack/range/delta*%
_class
loc:@lm/rnn/TensorArray*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0

+lm/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lm/rnn/TensorArraylm/rnn/TensorArrayStack/rangelm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
element_shape:	@
Y
lm/rnn/Const_2Const*
valueB:*
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
lm/rnn/range_1/startConst*
dtype0*
_output_shapes
: *
value	B :
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

lm/rnn/concat_2ConcatV2lm/rnn/concat_2/values_0lm/rnn/range_1lm/rnn/concat_2/axis*
_output_shapes
:*

Tidx0*
T0*
N
Ą
lm/rnn/transpose_1	Transpose+lm/rnn/TensorArrayStack/TensorArrayGatherV3lm/rnn/concat_2*
Tperm0*
T0*,
_output_shapes
:@˙˙˙˙˙˙˙˙˙
a
lm/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
|

lm/ReshapeReshapelm/rnn/transpose_1lm/Reshape/shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

*softmax_w/Initializer/random_uniform/shapeConst*
_class
loc:@softmax_w*
valueB"   Ţ  *
dtype0*
_output_shapes
:

(softmax_w/Initializer/random_uniform/minConst*
_class
loc:@softmax_w*
valueB
 *Yţź*
dtype0*
_output_shapes
: 

(softmax_w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@softmax_w*
valueB
 *Yţ<
Ţ
2softmax_w/Initializer/random_uniform/RandomUniformRandomUniform*softmax_w/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@softmax_w*
seed2 *
dtype0* 
_output_shapes
:
Ţ/
Â
(softmax_w/Initializer/random_uniform/subSub(softmax_w/Initializer/random_uniform/max(softmax_w/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_w*
_output_shapes
: 
Ö
(softmax_w/Initializer/random_uniform/mulMul2softmax_w/Initializer/random_uniform/RandomUniform(softmax_w/Initializer/random_uniform/sub* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@softmax_w
Č
$softmax_w/Initializer/random_uniformAdd(softmax_w/Initializer/random_uniform/mul(softmax_w/Initializer/random_uniform/min*
_class
loc:@softmax_w* 
_output_shapes
:
Ţ/*
T0

	softmax_w
VariableV2*
dtype0* 
_output_shapes
:
Ţ/*
shared_name *
_class
loc:@softmax_w*
	container *
shape:
Ţ/
˝
softmax_w/AssignAssign	softmax_w$softmax_w/Initializer/random_uniform* 
_output_shapes
:
Ţ/*
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
Ţ/

*softmax_b/Initializer/random_uniform/shapeConst*
_class
loc:@softmax_b*
valueB:Ţ/*
dtype0*
_output_shapes
:

(softmax_b/Initializer/random_uniform/minConst*
_class
loc:@softmax_b*
valueB
 *ľľź*
dtype0*
_output_shapes
: 

(softmax_b/Initializer/random_uniform/maxConst*
_class
loc:@softmax_b*
valueB
 *ľľ<*
dtype0*
_output_shapes
: 
Ů
2softmax_b/Initializer/random_uniform/RandomUniformRandomUniform*softmax_b/Initializer/random_uniform/shape*
dtype0*
_output_shapes	
:Ţ/*

seed *
T0*
_class
loc:@softmax_b*
seed2 
Â
(softmax_b/Initializer/random_uniform/subSub(softmax_b/Initializer/random_uniform/max(softmax_b/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_b*
_output_shapes
: 
Ń
(softmax_b/Initializer/random_uniform/mulMul2softmax_b/Initializer/random_uniform/RandomUniform(softmax_b/Initializer/random_uniform/sub*
_output_shapes	
:Ţ/*
T0*
_class
loc:@softmax_b
Ă
$softmax_b/Initializer/random_uniformAdd(softmax_b/Initializer/random_uniform/mul(softmax_b/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_b*
_output_shapes	
:Ţ/

	softmax_b
VariableV2*
shared_name *
_class
loc:@softmax_b*
	container *
shape:Ţ/*
dtype0*
_output_shapes	
:Ţ/
¸
softmax_b/AssignAssign	softmax_b$softmax_b/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@softmax_b*
validate_shape(*
_output_shapes	
:Ţ/
i
softmax_b/readIdentity	softmax_b*
T0*
_class
loc:@softmax_b*
_output_shapes	
:Ţ/

softmax/MatMulMatMul
lm/Reshapesoftmax_w/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/*
transpose_a( *
transpose_b( 

softmax/BiasAddBiasAddsoftmax/MatMulsoftmax_b/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/
`
Reshape/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
r
ReshapeReshapeinput/Placeholder_1Reshape/shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
one_hot/on_valueConst*
valueB
 *  ?*
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
B :Ţ/*
dtype0*
_output_shapes
: 
 
one_hotOneHotReshapeone_hot/depthone_hot/on_valueone_hot/off_value*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/*
T0*
axis˙˙˙˙˙˙˙˙˙*
TI0

>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientone_hot*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/*
T0
p
.loss/softmax_cross_entropy_with_logits_sg/RankConst*
dtype0*
_output_shapes
: *
value	B :
~
/loss/softmax_cross_entropy_with_logits_sg/ShapeShapesoftmax/BiasAdd*
_output_shapes
:*
T0*
out_type0
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

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
¸
-loss/softmax_cross_entropy_with_logits_sg/SubSub0loss/softmax_cross_entropy_with_logits_sg/Rank_1/loss/softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
Ś
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

/loss/softmax_cross_entropy_with_logits_sg/SliceSlice1loss/softmax_cross_entropy_with_logits_sg/Shape_15loss/softmax_cross_entropy_with_logits_sg/Slice/begin4loss/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:

9loss/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w
5loss/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

0loss/softmax_cross_entropy_with_logits_sg/concatConcatV29loss/softmax_cross_entropy_with_logits_sg/concat/values_0/loss/softmax_cross_entropy_with_logits_sg/Slice5loss/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Č
1loss/softmax_cross_entropy_with_logits_sg/ReshapeReshapesoftmax/BiasAdd0loss/softmax_cross_entropy_with_logits_sg/concat*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Ż
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
ź
/loss/softmax_cross_entropy_with_logits_sg/Sub_1Sub0loss/softmax_cross_entropy_with_logits_sg/Rank_21loss/softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
Ş
7loss/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack/loss/softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N*
_output_shapes
:

6loss/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

1loss/softmax_cross_entropy_with_logits_sg/Slice_1Slice1loss/softmax_cross_entropy_with_logits_sg/Shape_27loss/softmax_cross_entropy_with_logits_sg/Slice_1/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_1/size*
_output_shapes
:*
Index0*
T0

;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
7loss/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ą
2loss/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_01loss/softmax_cross_entropy_with_logits_sg/Slice_17loss/softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ű
3loss/softmax_cross_entropy_with_logits_sg/Reshape_1Reshape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient2loss/softmax_cross_entropy_with_logits_sg/concat_1*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
ü
)loss/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits1loss/softmax_cross_entropy_with_logits_sg/Reshape3loss/softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
ş
/loss/softmax_cross_entropy_with_logits_sg/Sub_2Sub.loss/softmax_cross_entropy_with_logits_sg/Rank1loss/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 

7loss/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Š
6loss/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack/loss/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:

1loss/softmax_cross_entropy_with_logits_sg/Slice_2Slice/loss/softmax_cross_entropy_with_logits_sg/Shape7loss/softmax_cross_entropy_with_logits_sg/Slice_2/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_2/size*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*
T0
Ř
3loss/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape)loss/softmax_cross_entropy_with_logits_sg1loss/softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

	loss/MeanMean3loss/softmax_cross_entropy_with_logits_sg/Reshape_2
loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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

train/gradients/ShapeShape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0
^
train/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
train/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
ś
train/gradients/f_count_1Entertrain/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context

train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
N*
_output_shapes
: : *
T0
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
Ř
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
Č
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context

train/gradients/Merge_1Mergetrain/gradients/b_count_1train/gradients/NextIteration_1*
N*
_output_shapes
: : *
T0

train/gradients/GreaterEqualGreaterEqualtrain/gradients/Merge_1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
Ď
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
train/gradients/Switch_1Switchtrain/gradients/Merge_1train/gradients/b_count_2*
T0*
_output_shapes
: : 
{
train/gradients/SubSubtrain/gradients/Switch_1:1"train/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
Ç
train/gradients/NextIteration_1NextIterationtrain/gradients/Sub_^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
\
train/gradients/b_count_3Exittrain/gradients/Switch_1*
T0*
_output_shapes
: 
ˇ
Ntrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape)loss/softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
ý
Ptrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapetrain/gradients/FillNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

train/gradients/zeros_like	ZerosLike+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Mtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ś
Itrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsPtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeMtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulMulItrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims+loss/softmax_cross_entropy_with_logits_sg:1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
Ĺ
Itrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax1loss/softmax_cross_entropy_with_logits_sg/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
Ď
Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/NegNegItrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Otrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ş
Ktrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsPtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeOtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1MulKtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Ltrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapesoftmax/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ź
Ntrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeBtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulLtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/
Ě
0train/gradients/softmax/BiasAdd_grad/BiasAddGradBiasAddGradNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes	
:Ţ/
í
*train/gradients/softmax/MatMul_grad/MatMulMatMulNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshapesoftmax_w/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ă
,train/gradients/softmax/MatMul_grad/MatMul_1MatMul
lm/ReshapeNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape* 
_output_shapes
:
Ţ/*
transpose_a(*
transpose_b( *
T0
w
%train/gradients/lm/Reshape_grad/ShapeShapelm/rnn/transpose_1*
T0*
out_type0*
_output_shapes
:
Ę
'train/gradients/lm/Reshape_grad/ReshapeReshape*train/gradients/softmax/MatMul_grad/MatMul%train/gradients/lm/Reshape_grad/Shape*
T0*
Tshape0*,
_output_shapes
:@˙˙˙˙˙˙˙˙˙

9train/gradients/lm/rnn/transpose_1_grad/InvertPermutationInvertPermutationlm/rnn/concat_2*
_output_shapes
:*
T0
ć
1train/gradients/lm/rnn/transpose_1_grad/transpose	Transpose'train/gradients/lm/Reshape_grad/Reshape9train/gradients/lm/rnn/transpose_1_grad/InvertPermutation*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0*
T0

btrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lm/rnn/TensorArraylm/rnn/while/Exit_2*
_output_shapes

:: *%
_class
loc:@lm/rnn/TensorArray*
sourcetrain/gradients
Ź
^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylm/rnn/while/Exit_2c^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*%
_class
loc:@lm/rnn/TensorArray*
_output_shapes
: 
ˇ
htrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3btrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3lm/rnn/TensorArrayStack/range1train/gradients/lm/rnn/transpose_1_grad/transpose^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
v
%train/gradients/zeros/shape_as_tensorConst*
valueB"@      *
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

train/gradients/zerosFill%train/gradients/zeros/shape_as_tensortrain/gradients/zeros/Const*
T0*

index_type0*
_output_shapes
:	@
x
'train/gradients/zeros_1/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"@      
b
train/gradients/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ł
train/gradients/zeros_1Fill'train/gradients/zeros_1/shape_as_tensortrain/gradients/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	@
x
'train/gradients/zeros_2/shape_as_tensorConst*
_output_shapes
:*
valueB"@      *
dtype0
b
train/gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ł
train/gradients/zeros_2Fill'train/gradients/zeros_2/shape_as_tensortrain/gradients/zeros_2/Const*
_output_shapes
:	@*
T0*

index_type0
x
'train/gradients/zeros_3/shape_as_tensorConst*
valueB"@      *
dtype0*
_output_shapes
:
b
train/gradients/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ł
train/gradients/zeros_3Fill'train/gradients/zeros_3/shape_as_tensortrain/gradients/zeros_3/Const*

index_type0*
_output_shapes
:	@*
T0
­
/train/gradients/lm/rnn/while/Exit_2_grad/b_exitEnterhtrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
ă
/train/gradients/lm/rnn/while/Exit_3_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@*:

frame_name,*train/gradients/lm/rnn/while/while_context
ĺ
/train/gradients/lm/rnn/while/Exit_4_grad/b_exitEntertrain/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@*:

frame_name,*train/gradients/lm/rnn/while/while_context
ĺ
/train/gradients/lm/rnn/while/Exit_5_grad/b_exitEntertrain/gradients/zeros_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@*:

frame_name,*train/gradients/lm/rnn/while/while_context
ĺ
/train/gradients/lm/rnn/while/Exit_6_grad/b_exitEntertrain/gradients/zeros_3*
_output_shapes
:	@*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
Ő
3train/gradients/lm/rnn/while/Switch_2_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_2_grad/b_exit:train/gradients/lm/rnn/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
Ţ
3train/gradients/lm/rnn/while/Switch_3_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_3_grad/b_exit:train/gradients/lm/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@: 
Ţ
3train/gradients/lm/rnn/while/Switch_4_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_4_grad/b_exit:train/gradients/lm/rnn/while/Switch_4_grad_1/NextIteration*!
_output_shapes
:	@: *
T0*
N
Ţ
3train/gradients/lm/rnn/while/Switch_5_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_5_grad/b_exit:train/gradients/lm/rnn/while/Switch_5_grad_1/NextIteration*
N*!
_output_shapes
:	@: *
T0
Ţ
3train/gradients/lm/rnn/while/Switch_6_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_6_grad/b_exit:train/gradients/lm/rnn/while/Switch_6_grad_1/NextIteration*
N*!
_output_shapes
:	@: *
T0
ő
0train/gradients/lm/rnn/while/Merge_2_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
_output_shapes
: : *
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_2_grad/b_switch

0train/gradients/lm/rnn/while/Merge_3_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_3_grad/b_switch**
_output_shapes
:	@:	@

0train/gradients/lm/rnn/while/Merge_4_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_4_grad/b_switchtrain/gradients/b_count_2**
_output_shapes
:	@:	@*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_4_grad/b_switch

0train/gradients/lm/rnn/while/Merge_5_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_5_grad/b_switchtrain/gradients/b_count_2**
_output_shapes
:	@:	@*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_5_grad/b_switch

0train/gradients/lm/rnn/while/Merge_6_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_6_grad/b_switchtrain/gradients/b_count_2*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_6_grad/b_switch**
_output_shapes
:	@:	@*
T0

.train/gradients/lm/rnn/while/Enter_2_grad/ExitExit0train/gradients/lm/rnn/while/Merge_2_grad/Switch*
T0*
_output_shapes
: 

.train/gradients/lm/rnn/while/Enter_3_grad/ExitExit0train/gradients/lm/rnn/while/Merge_3_grad/Switch*
T0*
_output_shapes
:	@

.train/gradients/lm/rnn/while/Enter_4_grad/ExitExit0train/gradients/lm/rnn/while/Merge_4_grad/Switch*
T0*
_output_shapes
:	@

.train/gradients/lm/rnn/while/Enter_5_grad/ExitExit0train/gradients/lm/rnn/while/Merge_5_grad/Switch*
_output_shapes
:	@*
T0

.train/gradients/lm/rnn/while/Enter_6_grad/ExitExit0train/gradients/lm/rnn/while/Merge_6_grad/Switch*
T0*
_output_shapes
:	@
Ť
gtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3mtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
sourcetrain/gradients*
_output_shapes

:: 
ę
mtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlm/rnn/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5
˙
ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1h^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
_output_shapes
: *
T0
Ě
Wtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3gtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
:	@*
dtype0
Ô
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
_output_shapes
: **
_class 
loc:@lm/rnn/while/Identity_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
˛
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const**
_class 
loc:@lm/rnn/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
Ä
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(
´
ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlm/rnn/while/Identity_1^train/gradients/Add*
_output_shapes
: *
swap_memory( *
T0

btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
ß
htrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ő
^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerc^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2i^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2g^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2_^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
¤
train/gradients/AddNAddN2train/gradients/lm/rnn/while/Merge_6_grad/Switch:1Wtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_6_grad/b_switch*
N*
_output_shapes
:	@
ţ
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/MulMultrain/gradients/AddN`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ű
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
×
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
â
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1Multrain/gradients/AddNbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
ú
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ř
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ă
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
Ě
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3_grad/TanhGradTanhGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul*
_output_shapes
:	@*
T0
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1*
T0*
_output_shapes
:	@
 
:train/gradients/lm/rnn/while/Switch_2_grad_1/NextIterationNextIteration2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1*
_output_shapes
: *
T0
Ş
train/gradients/AddN_1AddN2train/gradients/lm/rnn/while/Merge_5_grad/Switch:1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3_grad/TanhGrad*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_5_grad/b_switch*
N*
_output_shapes
:	@

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/MulMultrain/gradients/AddN_1`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ű
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/ConstConst*
_output_shapes
: *S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
×
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
â
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1Multrain/gradients/AddN_1btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
Ô
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/ConstConst**
_class 
loc:@lm/rnn/while/Identity_5*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
˛
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0**
_class 
loc:@lm/rnn/while/Identity_5
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
˝
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Enterlm/rnn/while/Identity_5^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/MulMultrain/gradients/AddN_1`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ř
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ô
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Const*
	elem_type0*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*

stack_name *
_output_shapes
:
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ß
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1Multrain/gradients/AddN_1btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
ý
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ű
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4*

stack_name *
_output_shapes
:*
	elem_type0
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ć
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1*
T0*
_output_shapes
:	@
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4_grad/SigmoidGradSigmoidGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul*
_output_shapes
:	@*
T0
Ě
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2_grad/TanhGradTanhGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1*
T0*
_output_shapes
:	@
Ě
:train/gradients/lm/rnn/while/Switch_5_grad_1/NextIterationNextIterationUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul*
T0*
_output_shapes
:	@
ž
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"@      
˛
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
ń
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ShapeYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
č
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/SumSumatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradgtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ě
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ReshapeReshapeUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/SumWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:	@
ě
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Sum_1Sumatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGraditrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
É
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Reshape_1ReshapeWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Sum_1Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concatConcatV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4_grad/SigmoidGrad[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2_grad/TanhGradYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Reshapeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5_grad/SigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat/Const*
T0*
N*
_output_shapes
:	@*

Tidx0
¸
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGradBiasAddGradZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat*
data_formatNHWC*
_output_shapes	
:*
T0
ô
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulMatMulZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concatatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/Enter*
_output_shapes
:	@*
transpose_a( *
transpose_b(*
T0
ś
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*:

frame_name,*train/gradients/lm/rnn/while/while_context
ţ
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1MatMulhtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/ConstConst*R
_classH
FDloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ć
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_accStackV2ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Const*
_output_shapes
:*
	elem_type0*R
_classH
FDloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1*

stack_name 
Đ
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/EnterEnterctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ń
itrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPushV2StackPushV2ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Enter?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
Ž
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2
StackPopV2ntrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ë
ntrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2/EnterEnterctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
˛
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
ą
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
ź
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/modFloorModZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConstYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Rank*
_output_shapes
: *
T0
Á
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ShapeConst^train/gradients/Sub*
valueB"@      *
dtype0*
_output_shapes
:
Ă
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1Const^train/gradients/Sub*
valueB"@      *
dtype0*
_output_shapes
:
°
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffsetConcatOffsetXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/modZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1*
N* 
_output_shapes
::
ś
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/SliceSlice[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffsetZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape*
Index0*
T0*
_output_shapes
:	@
ź
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice_1Slice[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffset:1\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1*
_output_shapes
:	@*
Index0*
T0
Š
train/gradients/AddN_2AddN2train/gradients/lm/rnn/while/Merge_4_grad/Switch:1Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_4_grad/b_switch*
N*
_output_shapes
:	@

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/MulMultrain/gradients/AddN_2`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ű
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
×
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
â
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1Multrain/gradients/AddN_2btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
ú
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ř
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ă
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ě
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul*
T0*
_output_shapes
:	@
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1*
T0*
_output_shapes
:	@
Ó
:train/gradients/lm/rnn/while/Switch_6_grad_1/NextIterationNextIteration\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice_1*
T0*
_output_shapes
:	@
Ş
train/gradients/AddN_3AddN2train/gradients/lm/rnn/while/Merge_3_grad/Switch:1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes
:	@
ü
Strain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/MulMultrain/gradients/AddN_3^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
÷
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/ConstConst*Q
_classG
ECloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ń
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Const*Q
_classG
ECloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
ź
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/EnterEnterYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Ü
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Enter>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2dtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@
×
dtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1Multrain/gradients/AddN_3`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
Ň
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/ConstConst**
_class 
loc:@lm/rnn/while/Identity_3*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ž
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Const**
_class 
loc:@lm/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
š
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Enterlm/rnn/while/Identity_3^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/MulMultrain/gradients/AddN_3`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
_output_shapes
:	@*
T0
ö
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*N
_classD
B@loc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ň
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Const*N
_classD
B@loc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
Ý
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Enter;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1Multrain/gradients/AddN_3btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
ý
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ű
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1*

stack_name 
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ć
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ď
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes
:	@*
T0
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul*
T0*
_output_shapes
:	@
Ę
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1*
T0*
_output_shapes
:	@
Ę
:train/gradients/lm/rnn/while/Switch_3_grad_1/NextIterationNextIterationStrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul*
T0*
_output_shapes
:	@
ź
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ShapeConst^train/gradients/Sub*
valueB"@      *
dtype0*
_output_shapes
:
°
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
ë
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ShapeWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
â
Strain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/SumSum_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradetrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ć
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ReshapeReshapeStrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/SumUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	@
ć
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Sum_1Sum_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradgtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ă
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Reshape_1ReshapeUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Sum_1Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ý
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concatConcatV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_grad/TanhGradWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Reshapeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes
:	@*

Tidx0
ś
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
đ
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMulXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concatatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/Enter*
T0*
_output_shapes
:	@*
transpose_a( *
transpose_b(
ř
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMulftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ţ
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ŕ
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
Ě
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnteratrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ë
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
Ş
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2ltrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ç
ltrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnteratrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 

train/gradients/AddN_4AddNatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad*
N*
_output_shapes	
:
Ž
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes	
:
Ű
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
ç
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Mergeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes
	:: *
T0

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*"
_output_shapes
::

]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAddbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1train/gradients/AddN_4*
T0*
_output_shapes	
:
ý
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
ń
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
°
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConstConst^train/gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
Ż
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
ś
Vtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/modFloorModXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConstWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
ż
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ShapeConst^train/gradients/Sub*
valueB"@      *
dtype0*
_output_shapes
:
Á
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"@      
¨
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetVtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/modXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ShapeZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::
Ž
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/SliceSliceYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape*
Index0*
T0*
_output_shapes
:	@
´
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Slice_1SliceYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1*
_output_shapes
:	@*
Index0*
T0

train/gradients/AddN_5AddN]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
*
T0*p
_classf
dbloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1
ˇ
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0* 
_output_shapes
:
*
valueB
*    
Ţ
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*
is_constant( *
parallel_iterations * 
_output_shapes
:
*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
é
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
: *
T0

_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*,
_output_shapes
:
:
*
T0

\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAddatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1train/gradients/AddN_5*
T0* 
_output_shapes
:


ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIteration\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
*
T0
ô
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:

°
Utrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3[train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^train/gradients/Sub*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
sourcetrain/gradients*
_output_shapes

:: 
Â
[train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlm/rnn/TensorArray_1*
parallel_iterations *
is_constant(*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter
í
]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
î
Qtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1V^train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter
ö
Wtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Utrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/SliceQtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0

Atrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterAtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context

Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Itrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
Ď
Btrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2train/gradients/b_count_2*
_output_shapes
: : *
T0

?train/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddDtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Wtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
ź
Itrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration?train/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
°
Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitBtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
Ń
:train/gradients/lm/rnn/while/Switch_4_grad_1/NextIterationNextIterationZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Slice_1*
T0*
_output_shapes
:	@
Ě
xtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lm/rnn/TensorArray_1Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *'
_class
loc:@lm/rnn/TensorArray_1*
sourcetrain/gradients

ttrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3y^train/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*'
_class
loc:@lm/rnn/TensorArray_1*
_output_shapes
: 
ä
jtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3xtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lm/rnn/TensorArrayUnstack/rangettrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@

7train/gradients/lm/rnn/transpose_grad/InvertPermutationInvertPermutationlm/rnn/concat*
T0*
_output_shapes
:
Ľ
/train/gradients/lm/rnn/transpose_grad/transpose	Transposejtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV37train/gradients/lm/rnn/transpose_grad/InvertPermutation*
T0*,
_output_shapes
:@˙˙˙˙˙˙˙˙˙*
Tperm0
Ľ
.train/gradients/lm/embedding_lookup_grad/ShapeConst*
_class
loc:@embedding*%
valueB	"Ţ             *
dtype0	*
_output_shapes
:
ş
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
Ú
3train/gradients/lm/embedding_lookup_grad/ExpandDims
ExpandDims-train/gradients/lm/embedding_lookup_grad/Size7train/gradients/lm/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:

<train/gradients/lm/embedding_lookup_grad/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:

>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ě
6train/gradients/lm/embedding_lookup_grad/strided_sliceStridedSlice0train/gradients/lm/embedding_lookup_grad/ToInt32<train/gradients/lm/embedding_lookup_grad/strided_slice/stack>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_1>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
v
4train/gradients/lm/embedding_lookup_grad/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 

/train/gradients/lm/embedding_lookup_grad/concatConcatV23train/gradients/lm/embedding_lookup_grad/ExpandDims6train/gradients/lm/embedding_lookup_grad/strided_slice4train/gradients/lm/embedding_lookup_grad/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ţ
0train/gradients/lm/embedding_lookup_grad/ReshapeReshape/train/gradients/lm/rnn/transpose_grad/transpose/train/gradients/lm/embedding_lookup_grad/concat*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
2train/gradients/lm/embedding_lookup_grad/Reshape_1Reshapeinput/Placeholder3train/gradients/lm/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
train/global_norm/L2LossL2Loss0train/gradients/lm/embedding_lookup_grad/Reshape*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape*
_output_shapes
: 

train/global_norm/L2Loss_1L2Loss`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *
T0

train/global_norm/L2Loss_2L2Lossatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: 
´
train/global_norm/L2Loss_3L2Loss,train/gradients/softmax/MatMul_grad/MatMul_1*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1*
_output_shapes
: 
ź
train/global_norm/L2Loss_4L2Loss0train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
ă
train/global_norm/stackPacktrain/global_norm/L2Losstrain/global_norm/L2Loss_1train/global_norm/L2Loss_2train/global_norm/L2Loss_3train/global_norm/L2Loss_4*
T0*

axis *
N*
_output_shapes
:
a
train/global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:

train/global_norm/SumSumtrain/global_norm/stacktrain/global_norm/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
^
train/global_norm/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
o
train/global_norm/mulMultrain/global_norm/Sumtrain/global_norm/Const_1*
T0*
_output_shapes
: 
]
train/global_norm/global_normSqrttrain/global_norm/mul*
_output_shapes
: *
T0
h
#train/clip_by_global_norm/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

!train/clip_by_global_norm/truedivRealDiv#train/clip_by_global_norm/truediv/xtrain/global_norm/global_norm*
T0*
_output_shapes
: 
d
train/clip_by_global_norm/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
%train/clip_by_global_norm/truediv_1/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0

#train/clip_by_global_norm/truediv_1RealDivtrain/clip_by_global_norm/Const%train/clip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 

!train/clip_by_global_norm/MinimumMinimum!train/clip_by_global_norm/truediv#train/clip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
d
train/clip_by_global_norm/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @

train/clip_by_global_norm/mulMultrain/clip_by_global_norm/mul/x!train/clip_by_global_norm/Minimum*
_output_shapes
: *
T0
ď
train/clip_by_global_norm/mul_1Mul0train/gradients/lm/embedding_lookup_grad/Reshapetrain/clip_by_global_norm/mul*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
6train/clip_by_global_norm/train/clip_by_global_norm/_0Identitytrain/clip_by_global_norm/mul_1*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
train/clip_by_global_norm/mul_2Mul`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3train/clip_by_global_norm/mul*
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:


6train/clip_by_global_norm/train/clip_by_global_norm/_1Identitytrain/clip_by_global_norm/mul_2*
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3* 
_output_shapes
:

Ä
train/clip_by_global_norm/mul_3Mulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3train/clip_by_global_norm/mul*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:
˙
6train/clip_by_global_norm/train/clip_by_global_norm/_2Identitytrain/clip_by_global_norm/mul_3*
_output_shapes	
:*
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
ß
train/clip_by_global_norm/mul_4Mul,train/gradients/softmax/MatMul_grad/MatMul_1train/clip_by_global_norm/mul*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1* 
_output_shapes
:
Ţ/
Ď
6train/clip_by_global_norm/train/clip_by_global_norm/_3Identitytrain/clip_by_global_norm/mul_4* 
_output_shapes
:
Ţ/*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1
â
train/clip_by_global_norm/mul_5Mul0train/gradients/softmax/BiasAdd_grad/BiasAddGradtrain/clip_by_global_norm/mul*
_output_shapes	
:Ţ/*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad
Î
6train/clip_by_global_norm/train/clip_by_global_norm/_4Identitytrain/clip_by_global_norm/mul_5*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:Ţ/

train/beta1_power/initial_valueConst*
_class
loc:@embedding*
valueB
 *fff?*
dtype0*
_output_shapes
: 

train/beta1_power
VariableV2*
shared_name *
_class
loc:@embedding*
	container *
shape: *
dtype0*
_output_shapes
: 
ž
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: *
use_locking(
t
train/beta1_power/readIdentitytrain/beta1_power*
T0*
_class
loc:@embedding*
_output_shapes
: 

train/beta2_power/initial_valueConst*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *wž?*
dtype0

train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@embedding*
	container *
shape: 
ž
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
t
train/beta2_power/readIdentitytrain/beta2_power*
T0*
_class
loc:@embedding*
_output_shapes
: 

0embedding/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@embedding*
valueB"Ţ     *
dtype0*
_output_shapes
:

&embedding/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *    
Ý
 embedding/Adam/Initializer/zerosFill0embedding/Adam/Initializer/zeros/shape_as_tensor&embedding/Adam/Initializer/zeros/Const*
_class
loc:@embedding*

index_type0* 
_output_shapes
:
Ţ/*
T0
¤
embedding/Adam
VariableV2*
shared_name *
_class
loc:@embedding*
	container *
shape:
Ţ/*
dtype0* 
_output_shapes
:
Ţ/
Ă
embedding/Adam/AssignAssignembedding/Adam embedding/Adam/Initializer/zeros*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/*
use_locking(*
T0
x
embedding/Adam/readIdentityembedding/Adam* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@embedding
Ą
2embedding/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@embedding*
valueB"Ţ     

(embedding/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
ă
"embedding/Adam_1/Initializer/zerosFill2embedding/Adam_1/Initializer/zeros/shape_as_tensor(embedding/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@embedding*

index_type0
Ś
embedding/Adam_1
VariableV2*
shape:
Ţ/*
dtype0* 
_output_shapes
:
Ţ/*
shared_name *
_class
loc:@embedding*
	container 
É
embedding/Adam_1/AssignAssignembedding/Adam_1"embedding/Adam_1/Initializer/zeros* 
_output_shapes
:
Ţ/*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(
|
embedding/Adam_1/readIdentityembedding/Adam_1* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@embedding
í
Wrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
×
Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *    
ů
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zerosFillWrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorMrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*

index_type0* 
_output_shapes
:

ň
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ß
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:

í
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:

ď
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
˙
Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*

index_type0* 
_output_shapes
:

ô
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ĺ
>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ń
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:

×
Ernn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB*    *
dtype0*
_output_shapes	
:
ä
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ň
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
â
8rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:
Ů
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB*    *
dtype0
ć
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container 
Ř
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ć
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:

0softmax_w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_w*
valueB"   Ţ  *
dtype0*
_output_shapes
:

&softmax_w/Adam/Initializer/zeros/ConstConst*
_class
loc:@softmax_w*
valueB
 *    *
dtype0*
_output_shapes
: 
Ý
 softmax_w/Adam/Initializer/zerosFill0softmax_w/Adam/Initializer/zeros/shape_as_tensor&softmax_w/Adam/Initializer/zeros/Const*
_class
loc:@softmax_w*

index_type0* 
_output_shapes
:
Ţ/*
T0
¤
softmax_w/Adam
VariableV2*
dtype0* 
_output_shapes
:
Ţ/*
shared_name *
_class
loc:@softmax_w*
	container *
shape:
Ţ/
Ă
softmax_w/Adam/AssignAssignsoftmax_w/Adam softmax_w/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@softmax_w*
validate_shape(* 
_output_shapes
:
Ţ/
x
softmax_w/Adam/readIdentitysoftmax_w/Adam*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
Ţ/
Ą
2softmax_w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_w*
valueB"   Ţ  *
dtype0*
_output_shapes
:

(softmax_w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@softmax_w*
valueB
 *    *
dtype0*
_output_shapes
: 
ă
"softmax_w/Adam_1/Initializer/zerosFill2softmax_w/Adam_1/Initializer/zeros/shape_as_tensor(softmax_w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@softmax_w*

index_type0* 
_output_shapes
:
Ţ/
Ś
softmax_w/Adam_1
VariableV2*
shared_name *
_class
loc:@softmax_w*
	container *
shape:
Ţ/*
dtype0* 
_output_shapes
:
Ţ/
É
softmax_w/Adam_1/AssignAssignsoftmax_w/Adam_1"softmax_w/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@softmax_w*
validate_shape(* 
_output_shapes
:
Ţ/
|
softmax_w/Adam_1/readIdentitysoftmax_w/Adam_1* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@softmax_w

0softmax_b/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_b*
valueB:Ţ/*
dtype0*
_output_shapes
:

&softmax_b/Adam/Initializer/zeros/ConstConst*
_class
loc:@softmax_b*
valueB
 *    *
dtype0*
_output_shapes
: 
Ř
 softmax_b/Adam/Initializer/zerosFill0softmax_b/Adam/Initializer/zeros/shape_as_tensor&softmax_b/Adam/Initializer/zeros/Const*
_output_shapes	
:Ţ/*
T0*
_class
loc:@softmax_b*

index_type0

softmax_b/Adam
VariableV2*
shared_name *
_class
loc:@softmax_b*
	container *
shape:Ţ/*
dtype0*
_output_shapes	
:Ţ/
ž
softmax_b/Adam/AssignAssignsoftmax_b/Adam softmax_b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:Ţ/*
use_locking(*
T0*
_class
loc:@softmax_b
s
softmax_b/Adam/readIdentitysoftmax_b/Adam*
T0*
_class
loc:@softmax_b*
_output_shapes	
:Ţ/

2softmax_b/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_b*
valueB:Ţ/*
dtype0*
_output_shapes
:

(softmax_b/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@softmax_b*
valueB
 *    *
dtype0*
_output_shapes
: 
Ţ
"softmax_b/Adam_1/Initializer/zerosFill2softmax_b/Adam_1/Initializer/zeros/shape_as_tensor(softmax_b/Adam_1/Initializer/zeros/Const*
_output_shapes	
:Ţ/*
T0*
_class
loc:@softmax_b*

index_type0

softmax_b/Adam_1
VariableV2*
dtype0*
_output_shapes	
:Ţ/*
shared_name *
_class
loc:@softmax_b*
	container *
shape:Ţ/
Ä
softmax_b/Adam_1/AssignAssignsoftmax_b/Adam_1"softmax_b/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:Ţ/*
use_locking(*
T0*
_class
loc:@softmax_b
w
softmax_b/Adam_1/readIdentitysoftmax_b/Adam_1*
_output_shapes	
:Ţ/*
T0*
_class
loc:@softmax_b
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
 *wž?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ę
"train/Adam/update_embedding/UniqueUnique2train/gradients/lm/embedding_lookup_grad/Reshape_1*
out_idx0*
T0*
_class
loc:@embedding*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ą
!train/Adam/update_embedding/ShapeShape"train/Adam/update_embedding/Unique*
_output_shapes
:*
T0*
_class
loc:@embedding*
out_type0

/train/Adam/update_embedding/strided_slice/stackConst*
_class
loc:@embedding*
valueB: *
dtype0*
_output_shapes
:

1train/Adam/update_embedding/strided_slice/stack_1Const*
_class
loc:@embedding*
valueB:*
dtype0*
_output_shapes
:

1train/Adam/update_embedding/strided_slice/stack_2Const*
_output_shapes
:*
_class
loc:@embedding*
valueB:*
dtype0
Ł
)train/Adam/update_embedding/strided_sliceStridedSlice!train/Adam/update_embedding/Shape/train/Adam/update_embedding/strided_slice/stack1train/Adam/update_embedding/strided_slice/stack_11train/Adam/update_embedding/strided_slice/stack_2*
T0*
Index0*
_class
loc:@embedding*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Â
.train/Adam/update_embedding/UnsortedSegmentSumUnsortedSegmentSum6train/clip_by_global_norm/train/clip_by_global_norm/_0$train/Adam/update_embedding/Unique:1)train/Adam/update_embedding/strided_slice*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tnumsegments0*
Tindices0*
T0*
_class
loc:@embedding

!train/Adam/update_embedding/sub/xConst*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
 
train/Adam/update_embedding/subSub!train/Adam/update_embedding/sub/xtrain/beta2_power/read*
T0*
_class
loc:@embedding*
_output_shapes
: 

 train/Adam/update_embedding/SqrtSqrttrain/Adam/update_embedding/sub*
T0*
_class
loc:@embedding*
_output_shapes
: 
¤
train/Adam/update_embedding/mulMullearning_rate/Variable/read train/Adam/update_embedding/Sqrt*
T0*
_class
loc:@embedding*
_output_shapes
: 

#train/Adam/update_embedding/sub_1/xConst*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *  ?*
dtype0
¤
!train/Adam/update_embedding/sub_1Sub#train/Adam/update_embedding/sub_1/xtrain/beta1_power/read*
T0*
_class
loc:@embedding*
_output_shapes
: 
ą
#train/Adam/update_embedding/truedivRealDivtrain/Adam/update_embedding/mul!train/Adam/update_embedding/sub_1*
T0*
_class
loc:@embedding*
_output_shapes
: 

#train/Adam/update_embedding/sub_2/xConst*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *  ?*
dtype0

!train/Adam/update_embedding/sub_2Sub#train/Adam/update_embedding/sub_2/xtrain/Adam/beta1*
_class
loc:@embedding*
_output_shapes
: *
T0
Ě
!train/Adam/update_embedding/mul_1Mul.train/Adam/update_embedding/UnsortedSegmentSum!train/Adam/update_embedding/sub_2*
T0*
_class
loc:@embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

!train/Adam/update_embedding/mul_2Mulembedding/Adam/readtrain/Adam/beta1*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Ń
"train/Adam/update_embedding/AssignAssignembedding/Adam!train/Adam/update_embedding/mul_2*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/

&train/Adam/update_embedding/ScatterAdd
ScatterAddembedding/Adam"train/Adam/update_embedding/Unique!train/Adam/update_embedding/mul_1#^train/Adam/update_embedding/Assign*
_class
loc:@embedding* 
_output_shapes
:
Ţ/*
use_locking( *
Tindices0*
T0
Ů
!train/Adam/update_embedding/mul_3Mul.train/Adam/update_embedding/UnsortedSegmentSum.train/Adam/update_embedding/UnsortedSegmentSum*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@embedding

#train/Adam/update_embedding/sub_3/xConst*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 

!train/Adam/update_embedding/sub_3Sub#train/Adam/update_embedding/sub_3/xtrain/Adam/beta2*
_class
loc:@embedding*
_output_shapes
: *
T0
ż
!train/Adam/update_embedding/mul_4Mul!train/Adam/update_embedding/mul_3!train/Adam/update_embedding/sub_3*
T0*
_class
loc:@embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

!train/Adam/update_embedding/mul_5Mulembedding/Adam_1/readtrain/Adam/beta2*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Ő
$train/Adam/update_embedding/Assign_1Assignembedding/Adam_1!train/Adam/update_embedding/mul_5*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/
˘
(train/Adam/update_embedding/ScatterAdd_1
ScatterAddembedding/Adam_1"train/Adam/update_embedding/Unique!train/Adam/update_embedding/mul_4%^train/Adam/update_embedding/Assign_1* 
_output_shapes
:
Ţ/*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding

"train/Adam/update_embedding/Sqrt_1Sqrt(train/Adam/update_embedding/ScatterAdd_1*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
ž
!train/Adam/update_embedding/mul_6Mul#train/Adam/update_embedding/truediv&train/Adam/update_embedding/ScatterAdd*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
§
train/Adam/update_embedding/addAdd"train/Adam/update_embedding/Sqrt_1train/Adam/epsilon* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@embedding
˝
%train/Adam/update_embedding/truediv_1RealDiv!train/Adam/update_embedding/mul_6train/Adam/update_embedding/add*
_class
loc:@embedding* 
_output_shapes
:
Ţ/*
T0
Ŕ
%train/Adam/update_embedding/AssignSub	AssignSub	embedding%train/Adam/update_embedding/truediv_1*
use_locking( *
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Č
&train/Adam/update_embedding/group_depsNoOp&^train/Adam/update_embedding/AssignSub'^train/Adam/update_embedding/ScatterAdd)^train/Adam/update_embedding/ScatterAdd_1*
_class
loc:@embedding
Ď
Ltrain/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_1*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
Ŕ
Jtrain/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_2*
use_locking( *
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes	
:

%train/Adam/update_softmax_w/ApplyAdam	ApplyAdam	softmax_wsoftmax_w/Adamsoftmax_w/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_3*
use_locking( *
T0*
_class
loc:@softmax_w*
use_nesterov( * 
_output_shapes
:
Ţ/

%train/Adam/update_softmax_b/ApplyAdam	ApplyAdam	softmax_bsoftmax_b/Adamsoftmax_b/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_4*
use_locking( *
T0*
_class
loc:@softmax_b*
use_nesterov( *
_output_shapes	
:Ţ/

train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam*
T0*
_class
loc:@embedding*
_output_shapes
: 
Ś
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 

train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam*
T0*
_class
loc:@embedding*
_output_shapes
: 
Ş
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
Ń

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam

initNoOp^embedding/Adam/Assign^embedding/Adam_1/Assign^embedding/Assign^learning_rate/Variable/Assign;^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign6^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign8^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign^softmax_b/Adam/Assign^softmax_b/Adam_1/Assign^softmax_b/Assign^softmax_w/Adam/Assign^softmax_w/Adam_1/Assign^softmax_w/Assign^train/beta1_power/Assign^train/beta2_power/Assign
\
Merge/MergeSummaryMergeSummarylearning_rate_1loss_1*
N*
_output_shapes
: ".Nä<Ă     )˙Ş	´ă:ŢĘÖAJŻ
5ĺ4
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
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignSub
ref"T

value"T

output_ref"T" 
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

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

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

2	

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

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint˙˙˙˙˙˙˙˙˙"	
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
2	
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
Ľ

ScatterAdd
ref"T
indices"Tindices
updates"T

output_ref"T" 
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
	elem_typetype
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring 
2
StopGradient

input"T
output"T"	
Ttype
ö
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

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
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
Ţ
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
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
Á
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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.8.02v1.8.0-0-g93bc2e2072ŔČ	
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:@˙˙˙˙˙˙˙˙˙*
shape:@˙˙˙˙˙˙˙˙˙
v
input/Placeholder_1Placeholder*
dtype0*'
_output_shapes
:@˙˙˙˙˙˙˙˙˙*
shape:@˙˙˙˙˙˙˙˙˙
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
Ú
learning_rate/Variable/AssignAssignlearning_rate/Variable$learning_rate/Variable/initial_value*
use_locking(*
T0*)
_class
loc:@learning_rate/Variable*
validate_shape(*
_output_shapes
: 

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
learning_rate_1ScalarSummarylearning_rate_1/tagslearning_rate/Variable/read*
_output_shapes
: *
T0

;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:@*
dtype0*
_output_shapes
:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:

Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Á
<rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concatConcatV2;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_1Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:

Arnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zerosFill<rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concatArnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes
:	@

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_2Const*
_output_shapes
:*
valueB:@*
dtype0

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_4Const*
dtype0*
_output_shapes
:*
valueB:@

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ç
>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1ConcatV2=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_4=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_5Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0

Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1Fill>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/Const*

index_type0*
_output_shapes
:	@*
T0

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:@*
dtype0*
_output_shapes
:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ConstConst*
valueB:@*
dtype0*
_output_shapes
:

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
É
>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatConcatV2=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_1Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

Crnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zerosFill>rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatCrnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/Const*
_output_shapes
:	@*
T0*

index_type0

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_2Const*
_output_shapes
:*
valueB:@*
dtype0

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_3Const*
_output_shapes
:*
valueB:*
dtype0

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_4Const*
valueB:@*
dtype0*
_output_shapes
:

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ď
@rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1ConcatV2?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_4?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_5Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:

Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1Fill@rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1Ernn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	@

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_6Const*
dtype0*
_output_shapes
:*
valueB:@

?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

*embedding/Initializer/random_uniform/shapeConst*
_class
loc:@embedding*
valueB"Ţ     *
dtype0*
_output_shapes
:

(embedding/Initializer/random_uniform/minConst*
_class
loc:@embedding*
valueB
 *Yţź*
dtype0*
_output_shapes
: 

(embedding/Initializer/random_uniform/maxConst*
_class
loc:@embedding*
valueB
 *Yţ<*
dtype0*
_output_shapes
: 
Ţ
2embedding/Initializer/random_uniform/RandomUniformRandomUniform*embedding/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
Ţ/*

seed *
T0*
_class
loc:@embedding*
seed2 
Â
(embedding/Initializer/random_uniform/subSub(embedding/Initializer/random_uniform/max(embedding/Initializer/random_uniform/min*
T0*
_class
loc:@embedding*
_output_shapes
: 
Ö
(embedding/Initializer/random_uniform/mulMul2embedding/Initializer/random_uniform/RandomUniform(embedding/Initializer/random_uniform/sub* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@embedding
Č
$embedding/Initializer/random_uniformAdd(embedding/Initializer/random_uniform/mul(embedding/Initializer/random_uniform/min*
_class
loc:@embedding* 
_output_shapes
:
Ţ/*
T0

	embedding
VariableV2* 
_output_shapes
:
Ţ/*
shared_name *
_class
loc:@embedding*
	container *
shape:
Ţ/*
dtype0
˝
embedding/AssignAssign	embedding$embedding/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/
n
embedding/readIdentity	embedding*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
x
lm/embedding_lookup/axisConst*
_class
loc:@embedding*
value	B : *
dtype0*
_output_shapes
: 
Ô
lm/embedding_lookupGatherV2embedding/readinput/Placeholderlm/embedding_lookup/axis*,
_output_shapes
:@˙˙˙˙˙˙˙˙˙*
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

lm/rnn/concatConcatV2lm/rnn/concat/values_0lm/rnn/rangelm/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

lm/rnn/transpose	Transposelm/embedding_lookuplm/rnn/concat*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0*
T0
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

lm/rnn/strided_sliceStridedSlicelm/rnn/Shapelm/rnn/strided_slice/stacklm/rnn/strided_slice/stack_1lm/rnn/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
V
lm/rnn/ConstConst*
valueB:@*
dtype0*
_output_shapes
:
Y
lm/rnn/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
V
lm/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

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
:	@
M
lm/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 

lm/rnn/TensorArrayTensorArrayV3lm/rnn/strided_slice*2
tensor_array_namelm/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:	@*
dynamic_size( *
clear_after_read(*
identical_element_shapes(

lm/rnn/TensorArray_1TensorArrayV3lm/rnn/strided_slice*1
tensor_array_namelm/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:	@*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
o
lm/rnn/TensorArrayUnstack/ShapeShapelm/rnn/transpose*
T0*
out_type0*
_output_shapes
:
w
-lm/rnn/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
y
/lm/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/lm/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ű
'lm/rnn/TensorArrayUnstack/strided_sliceStridedSlicelm/rnn/TensorArrayUnstack/Shape-lm/rnn/TensorArrayUnstack/strided_slice/stack/lm/rnn/TensorArrayUnstack/strided_slice/stack_1/lm/rnn/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
g
%lm/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%lm/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Đ
lm/rnn/TensorArrayUnstack/rangeRange%lm/rnn/TensorArrayUnstack/range/start'lm/rnn/TensorArrayUnstack/strided_slice%lm/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0

Alm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lm/rnn/TensorArray_1lm/rnn/TensorArrayUnstack/rangelm/rnn/transposelm/rnn/TensorArray_1:1*
_output_shapes
: *
T0*#
_class
loc:@lm/rnn/transpose
R
lm/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
b
lm/rnn/MaximumMaximumlm/rnn/Maximum/xlm/rnn/strided_slice*
T0*
_output_shapes
: 
`
lm/rnn/MinimumMinimumlm/rnn/strided_slicelm/rnn/Maximum*
_output_shapes
: *
T0
`
lm/rnn/while/iteration_counterConst*
_output_shapes
: *
value	B : *
dtype0
ś
lm/rnn/while/EnterEnterlm/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
Ľ
lm/rnn/while/Enter_1Enterlm/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
Ž
lm/rnn/while/Enter_2Enterlm/rnn/TensorArray:1*
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
Ţ
lm/rnn/while/Enter_3Enter;rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@**

frame_namelm/rnn/while/while_context
ŕ
lm/rnn/while/Enter_4Enter=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1*
_output_shapes
:	@**

frame_namelm/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
ŕ
lm/rnn/while/Enter_5Enter=rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@**

frame_namelm/rnn/while/while_context
â
lm/rnn/while/Enter_6Enter?rnn_lstm/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1*
is_constant( *
parallel_iterations *
_output_shapes
:	@**

frame_namelm/rnn/while/while_context*
T0
w
lm/rnn/while/MergeMergelm/rnn/while/Enterlm/rnn/while/NextIteration*
_output_shapes
: : *
T0*
N
}
lm/rnn/while/Merge_1Mergelm/rnn/while/Enter_1lm/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
}
lm/rnn/while/Merge_2Mergelm/rnn/while/Enter_2lm/rnn/while/NextIteration_2*
N*
_output_shapes
: : *
T0

lm/rnn/while/Merge_3Mergelm/rnn/while/Enter_3lm/rnn/while/NextIteration_3*
N*!
_output_shapes
:	@: *
T0

lm/rnn/while/Merge_4Mergelm/rnn/while/Enter_4lm/rnn/while/NextIteration_4*
T0*
N*!
_output_shapes
:	@: 

lm/rnn/while/Merge_5Mergelm/rnn/while/Enter_5lm/rnn/while/NextIteration_5*
N*!
_output_shapes
:	@: *
T0

lm/rnn/while/Merge_6Mergelm/rnn/while/Enter_6lm/rnn/while/NextIteration_6*
T0*
N*!
_output_shapes
:	@: 
g
lm/rnn/while/LessLesslm/rnn/while/Mergelm/rnn/while/Less/Enter*
T0*
_output_shapes
: 
ą
lm/rnn/while/Less/EnterEnterlm/rnn/strided_slice*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context
m
lm/rnn/while/Less_1Lesslm/rnn/while/Merge_1lm/rnn/while/Less_1/Enter*
_output_shapes
: *
T0
­
lm/rnn/while/Less_1/EnterEnterlm/rnn/Minimum*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0
e
lm/rnn/while/LogicalAnd
LogicalAndlm/rnn/while/Lesslm/rnn/while/Less_1*
_output_shapes
: 
R
lm/rnn/while/LoopCondLoopCondlm/rnn/while/LogicalAnd*
_output_shapes
: 

lm/rnn/while/SwitchSwitchlm/rnn/while/Mergelm/rnn/while/LoopCond*
T0*%
_class
loc:@lm/rnn/while/Merge*
_output_shapes
: : 

lm/rnn/while/Switch_1Switchlm/rnn/while/Merge_1lm/rnn/while/LoopCond*'
_class
loc:@lm/rnn/while/Merge_1*
_output_shapes
: : *
T0

lm/rnn/while/Switch_2Switchlm/rnn/while/Merge_2lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_2*
_output_shapes
: : 
Ş
lm/rnn/while/Switch_3Switchlm/rnn/while/Merge_3lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_3**
_output_shapes
:	@:	@
Ş
lm/rnn/while/Switch_4Switchlm/rnn/while/Merge_4lm/rnn/while/LoopCond*
T0*'
_class
loc:@lm/rnn/while/Merge_4**
_output_shapes
:	@:	@
Ş
lm/rnn/while/Switch_5Switchlm/rnn/while/Merge_5lm/rnn/while/LoopCond**
_output_shapes
:	@:	@*
T0*'
_class
loc:@lm/rnn/while/Merge_5
Ş
lm/rnn/while/Switch_6Switchlm/rnn/while/Merge_6lm/rnn/while/LoopCond*'
_class
loc:@lm/rnn/while/Merge_6**
_output_shapes
:	@:	@*
T0
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
lm/rnn/while/Identity_3Identitylm/rnn/while/Switch_3:1*
_output_shapes
:	@*
T0
f
lm/rnn/while/Identity_4Identitylm/rnn/while/Switch_4:1*
T0*
_output_shapes
:	@
f
lm/rnn/while/Identity_5Identitylm/rnn/while/Switch_5:1*
T0*
_output_shapes
:	@
f
lm/rnn/while/Identity_6Identitylm/rnn/while/Switch_6:1*
_output_shapes
:	@*
T0
l
lm/rnn/while/add/yConst^lm/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
c
lm/rnn/while/addAddlm/rnn/while/Identitylm/rnn/while/add/y*
_output_shapes
: *
T0
Č
lm/rnn/while/TensorArrayReadV3TensorArrayReadV3$lm/rnn/while/TensorArrayReadV3/Enterlm/rnn/while/Identity_1&lm/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	@
Â
$lm/rnn/while/TensorArrayReadV3/EnterEnterlm/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
í
&lm/rnn/while/TensorArrayReadV3/Enter_1EnterAlm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0
ç
Qrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *óľ˝*
dtype0*
_output_shapes
: 
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *óľ=*
dtype0*
_output_shapes
: 
Ó
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
seed2 
Ţ
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: 
ň
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:

ä
Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
*
T0
í
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:

Ů
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:


5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0* 
_output_shapes
:

Ň
@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB*    *
dtype0*
_output_shapes	
:
ß
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ă
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:

3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
_output_shapes	
:

<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/ConstConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axisConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concatConcatV2lm/rnn/while/TensorArrayReadV3lm/rnn/while/Identity_4Blm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat/axis*
N*
_output_shapes
:	@*

Tidx0*
T0

=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMulMatMul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concatClm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter*
_output_shapes
:	@*
transpose_a( *
transpose_b( *
T0

Clm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
**

frame_namelm/rnn/while/while_context

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAddBiasAdd=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMulDlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	@

Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:**

frame_namelm/rnn/while/while_context

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_1Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/splitSplit<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd*
T0*@
_output_shapes.
,:	@:	@:	@:	@*
	num_split

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2Const^lm/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ë
:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/AddAdd>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:2>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	@
Ż
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/SigmoidSigmoid:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add*
T0*
_output_shapes
:	@
Ä
:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MulMullm/rnn/while/Identity_3>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes
:	@*
T0
ł
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split*
_output_shapes
:	@*
T0
­
;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/TanhTanh>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:1*
T0*
_output_shapes
:	@
ě
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1Mul@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	@
ç
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1Add:lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	@
­
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1Tanh<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1*
_output_shapes
:	@*
T0
ľ
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2Sigmoid>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split:3*
_output_shapes
:	@*
T0
î
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2Mul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	@

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axisConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
§
?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1ConcatV2<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2lm/rnn/while/Identity_6Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:	@

?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1MatMul?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1Clm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter*
_output_shapes
:	@*
transpose_a( *
transpose_b( *
T0

@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1BiasAdd?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1Dlm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	@*
T0

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_4Const^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ľ
>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1Split>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_3@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1*@
_output_shapes.
,:	@:	@:	@:	@*
	num_split*
T0

>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5Const^lm/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ď
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2Add@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:2>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Const_5*
T0*
_output_shapes
:	@
ł
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3Sigmoid<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2*
_output_shapes
:	@*
T0
Č
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3Mullm/rnn/while/Identity_5@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*
T0*
_output_shapes
:	@
ľ
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4Sigmoid>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1*
T0*
_output_shapes
:	@
ą
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2Tanh@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:1*
T0*
_output_shapes
:	@
î
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4Mul@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*
_output_shapes
:	@*
T0
é
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3Add<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4*
_output_shapes
:	@*
T0
­
=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3Tanh<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3*
_output_shapes
:	@*
T0
ˇ
@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5Sigmoid@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1:3*
_output_shapes
:	@*
T0
î
<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5Mul=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*
T0*
_output_shapes
:	@
ŕ
0lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterlm/rnn/while/Identity_1<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5lm/rnn/while/Identity_2*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
_output_shapes
: 
Ł
6lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlm/rnn/TensorArray*
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
parallel_iterations *
is_constant(
n
lm/rnn/while/add_1/yConst^lm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
lm/rnn/while/add_1Addlm/rnn/while/Identity_1lm/rnn/while/add_1/y*
T0*
_output_shapes
: 
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

lm/rnn/while/NextIteration_2NextIteration0lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

lm/rnn/while/NextIteration_3NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	@

lm/rnn/while/NextIteration_4NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2*
_output_shapes
:	@*
T0

lm/rnn/while/NextIteration_5NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_3*
T0*
_output_shapes
:	@

lm/rnn/while/NextIteration_6NextIteration<lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
T0*
_output_shapes
:	@
O
lm/rnn/while/ExitExitlm/rnn/while/Switch*
_output_shapes
: *
T0
S
lm/rnn/while/Exit_1Exitlm/rnn/while/Switch_1*
T0*
_output_shapes
: 
S
lm/rnn/while/Exit_2Exitlm/rnn/while/Switch_2*
_output_shapes
: *
T0
\
lm/rnn/while/Exit_3Exitlm/rnn/while/Switch_3*
_output_shapes
:	@*
T0
\
lm/rnn/while/Exit_4Exitlm/rnn/while/Switch_4*
T0*
_output_shapes
:	@
\
lm/rnn/while/Exit_5Exitlm/rnn/while/Switch_5*
T0*
_output_shapes
:	@
\
lm/rnn/while/Exit_6Exitlm/rnn/while/Switch_6*
T0*
_output_shapes
:	@
Ś
)lm/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lm/rnn/TensorArraylm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
_output_shapes
: 

#lm/rnn/TensorArrayStack/range/startConst*%
_class
loc:@lm/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

#lm/rnn/TensorArrayStack/range/deltaConst*%
_class
loc:@lm/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
ó
lm/rnn/TensorArrayStack/rangeRange#lm/rnn/TensorArrayStack/range/start)lm/rnn/TensorArrayStack/TensorArraySizeV3#lm/rnn/TensorArrayStack/range/delta*

Tidx0*%
_class
loc:@lm/rnn/TensorArray*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

+lm/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lm/rnn/TensorArraylm/rnn/TensorArrayStack/rangelm/rnn/while/Exit_2*
element_shape:	@*%
_class
loc:@lm/rnn/TensorArray*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Y
lm/rnn/Const_2Const*
valueB:*
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

lm/rnn/concat_2ConcatV2lm/rnn/concat_2/values_0lm/rnn/range_1lm/rnn/concat_2/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ą
lm/rnn/transpose_1	Transpose+lm/rnn/TensorArrayStack/TensorArrayGatherV3lm/rnn/concat_2*,
_output_shapes
:@˙˙˙˙˙˙˙˙˙*
Tperm0*
T0
a
lm/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
|

lm/ReshapeReshapelm/rnn/transpose_1lm/Reshape/shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

*softmax_w/Initializer/random_uniform/shapeConst*
_class
loc:@softmax_w*
valueB"   Ţ  *
dtype0*
_output_shapes
:

(softmax_w/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@softmax_w*
valueB
 *Yţź*
dtype0

(softmax_w/Initializer/random_uniform/maxConst*
_class
loc:@softmax_w*
valueB
 *Yţ<*
dtype0*
_output_shapes
: 
Ţ
2softmax_w/Initializer/random_uniform/RandomUniformRandomUniform*softmax_w/Initializer/random_uniform/shape* 
_output_shapes
:
Ţ/*

seed *
T0*
_class
loc:@softmax_w*
seed2 *
dtype0
Â
(softmax_w/Initializer/random_uniform/subSub(softmax_w/Initializer/random_uniform/max(softmax_w/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_w*
_output_shapes
: 
Ö
(softmax_w/Initializer/random_uniform/mulMul2softmax_w/Initializer/random_uniform/RandomUniform(softmax_w/Initializer/random_uniform/sub* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@softmax_w
Č
$softmax_w/Initializer/random_uniformAdd(softmax_w/Initializer/random_uniform/mul(softmax_w/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_w* 
_output_shapes
:
Ţ/

	softmax_w
VariableV2*
shared_name *
_class
loc:@softmax_w*
	container *
shape:
Ţ/*
dtype0* 
_output_shapes
:
Ţ/
˝
softmax_w/AssignAssign	softmax_w$softmax_w/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@softmax_w*
validate_shape(* 
_output_shapes
:
Ţ/
n
softmax_w/readIdentity	softmax_w*
_class
loc:@softmax_w* 
_output_shapes
:
Ţ/*
T0

*softmax_b/Initializer/random_uniform/shapeConst*
_class
loc:@softmax_b*
valueB:Ţ/*
dtype0*
_output_shapes
:

(softmax_b/Initializer/random_uniform/minConst*
_class
loc:@softmax_b*
valueB
 *ľľź*
dtype0*
_output_shapes
: 

(softmax_b/Initializer/random_uniform/maxConst*
_class
loc:@softmax_b*
valueB
 *ľľ<*
dtype0*
_output_shapes
: 
Ů
2softmax_b/Initializer/random_uniform/RandomUniformRandomUniform*softmax_b/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@softmax_b*
seed2 *
dtype0*
_output_shapes	
:Ţ/
Â
(softmax_b/Initializer/random_uniform/subSub(softmax_b/Initializer/random_uniform/max(softmax_b/Initializer/random_uniform/min*
T0*
_class
loc:@softmax_b*
_output_shapes
: 
Ń
(softmax_b/Initializer/random_uniform/mulMul2softmax_b/Initializer/random_uniform/RandomUniform(softmax_b/Initializer/random_uniform/sub*
T0*
_class
loc:@softmax_b*
_output_shapes	
:Ţ/
Ă
$softmax_b/Initializer/random_uniformAdd(softmax_b/Initializer/random_uniform/mul(softmax_b/Initializer/random_uniform/min*
_class
loc:@softmax_b*
_output_shapes	
:Ţ/*
T0

	softmax_b
VariableV2*
_output_shapes	
:Ţ/*
shared_name *
_class
loc:@softmax_b*
	container *
shape:Ţ/*
dtype0
¸
softmax_b/AssignAssign	softmax_b$softmax_b/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@softmax_b*
validate_shape(*
_output_shapes	
:Ţ/
i
softmax_b/readIdentity	softmax_b*
T0*
_class
loc:@softmax_b*
_output_shapes	
:Ţ/

softmax/MatMulMatMul
lm/Reshapesoftmax_w/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/*
transpose_a( *
transpose_b( 

softmax/BiasAddBiasAddsoftmax/MatMulsoftmax_b/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/
`
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
r
ReshapeReshapeinput/Placeholder_1Reshape/shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
one_hot/on_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
V
one_hot/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
P
one_hot/depthConst*
value
B :Ţ/*
dtype0*
_output_shapes
: 
 
one_hotOneHotReshapeone_hot/depthone_hot/on_valueone_hot/off_value*
T0*
axis˙˙˙˙˙˙˙˙˙*
TI0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/

>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientone_hot*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/
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
0loss/softmax_cross_entropy_with_logits_sg/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :

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
¸
-loss/softmax_cross_entropy_with_logits_sg/SubSub0loss/softmax_cross_entropy_with_logits_sg/Rank_1/loss/softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
Ś
5loss/softmax_cross_entropy_with_logits_sg/Slice/beginPack-loss/softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
~
4loss/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0

/loss/softmax_cross_entropy_with_logits_sg/SliceSlice1loss/softmax_cross_entropy_with_logits_sg/Shape_15loss/softmax_cross_entropy_with_logits_sg/Slice/begin4loss/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:

9loss/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w
5loss/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

0loss/softmax_cross_entropy_with_logits_sg/concatConcatV29loss/softmax_cross_entropy_with_logits_sg/concat/values_0/loss/softmax_cross_entropy_with_logits_sg/Slice5loss/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Č
1loss/softmax_cross_entropy_with_logits_sg/ReshapeReshapesoftmax/BiasAdd0loss/softmax_cross_entropy_with_logits_sg/concat*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Ż
1loss/softmax_cross_entropy_with_logits_sg/Shape_2Shape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
ź
/loss/softmax_cross_entropy_with_logits_sg/Sub_1Sub0loss/softmax_cross_entropy_with_logits_sg/Rank_21loss/softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
Ş
7loss/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack/loss/softmax_cross_entropy_with_logits_sg/Sub_1*
_output_shapes
:*
T0*

axis *
N

6loss/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

1loss/softmax_cross_entropy_with_logits_sg/Slice_1Slice1loss/softmax_cross_entropy_with_logits_sg/Shape_27loss/softmax_cross_entropy_with_logits_sg/Slice_1/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:

;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
y
7loss/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ą
2loss/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_01loss/softmax_cross_entropy_with_logits_sg/Slice_17loss/softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
ű
3loss/softmax_cross_entropy_with_logits_sg/Reshape_1Reshape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient2loss/softmax_cross_entropy_with_logits_sg/concat_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ü
)loss/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits1loss/softmax_cross_entropy_with_logits_sg/Reshape3loss/softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
ş
/loss/softmax_cross_entropy_with_logits_sg/Sub_2Sub.loss/softmax_cross_entropy_with_logits_sg/Rank1loss/softmax_cross_entropy_with_logits_sg/Sub_2/y*
_output_shapes
: *
T0

7loss/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Š
6loss/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack/loss/softmax_cross_entropy_with_logits_sg/Sub_2*
_output_shapes
:*
T0*

axis *
N

1loss/softmax_cross_entropy_with_logits_sg/Slice_2Slice/loss/softmax_cross_entropy_with_logits_sg/Shape7loss/softmax_cross_entropy_with_logits_sg/Slice_2/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
3loss/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape)loss/softmax_cross_entropy_with_logits_sg1loss/softmax_cross_entropy_with_logits_sg/Slice_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
T

loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0

	loss/MeanMean3loss/softmax_cross_entropy_with_logits_sg/Reshape_2
loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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

train/gradients/ShapeShape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
^
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?

train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
train/gradients/f_countConst*
dtype0*
_output_shapes
: *
value	B : 
ś
train/gradients/f_count_1Entertrain/gradients/f_count*
is_constant( *
parallel_iterations *
_output_shapes
: **

frame_namelm/rnn/while/while_context*
T0

train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
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
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
T0*
_output_shapes
: 
Ř
train/gradients/NextIterationNextIterationtrain/gradients/Addd^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2j^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPushV2h^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPushV2d^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPushV2`^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPushV2b^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
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
Č
train/gradients/b_count_1Entertrain/gradients/f_count_2*
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 

train/gradients/Merge_1Mergetrain/gradients/b_count_1train/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 

train/gradients/GreaterEqualGreaterEqualtrain/gradients/Merge_1"train/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
Ď
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
train/gradients/SubSubtrain/gradients/Switch_1:1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
Ç
train/gradients/NextIteration_1NextIterationtrain/gradients/Sub_^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
\
train/gradients/b_count_3Exittrain/gradients/Switch_1*
T0*
_output_shapes
: 
ˇ
Ntrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape)loss/softmax_cross_entropy_with_logits_sg*
_output_shapes
:*
T0*
out_type0
ý
Ptrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapetrain/gradients/FillNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

train/gradients/zeros_like	ZerosLike+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Mtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ś
Itrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsPtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeMtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulMulItrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ĺ
Itrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax1loss/softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ď
Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/NegNegItrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Otrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
ş
Ktrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsPtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeOtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0

Dtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1MulKtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Btrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/Neg*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

Ltrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapesoftmax/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ź
Ntrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeBtrain/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulLtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ/
Ě
0train/gradients/softmax/BiasAdd_grad/BiasAddGradBiasAddGradNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes	
:Ţ/
í
*train/gradients/softmax/MatMul_grad/MatMulMatMulNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshapesoftmax_w/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ă
,train/gradients/softmax/MatMul_grad/MatMul_1MatMul
lm/ReshapeNtrain/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0* 
_output_shapes
:
Ţ/*
transpose_a(*
transpose_b( 
w
%train/gradients/lm/Reshape_grad/ShapeShapelm/rnn/transpose_1*
out_type0*
_output_shapes
:*
T0
Ę
'train/gradients/lm/Reshape_grad/ReshapeReshape*train/gradients/softmax/MatMul_grad/MatMul%train/gradients/lm/Reshape_grad/Shape*
T0*
Tshape0*,
_output_shapes
:@˙˙˙˙˙˙˙˙˙

9train/gradients/lm/rnn/transpose_1_grad/InvertPermutationInvertPermutationlm/rnn/concat_2*
T0*
_output_shapes
:
ć
1train/gradients/lm/rnn/transpose_1_grad/transpose	Transpose'train/gradients/lm/Reshape_grad/Reshape9train/gradients/lm/rnn/transpose_1_grad/InvertPermutation*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
Tperm0

btrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lm/rnn/TensorArraylm/rnn/while/Exit_2*%
_class
loc:@lm/rnn/TensorArray*
sourcetrain/gradients*
_output_shapes

:: 
Ź
^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylm/rnn/while/Exit_2c^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*%
_class
loc:@lm/rnn/TensorArray*
_output_shapes
: 
ˇ
htrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3btrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3lm/rnn/TensorArrayStack/range1train/gradients/lm/rnn/transpose_1_grad/transpose^train/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
v
%train/gradients/zeros/shape_as_tensorConst*
valueB"@      *
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

train/gradients/zerosFill%train/gradients/zeros/shape_as_tensortrain/gradients/zeros/Const*
T0*

index_type0*
_output_shapes
:	@
x
'train/gradients/zeros_1/shape_as_tensorConst*
valueB"@      *
dtype0*
_output_shapes
:
b
train/gradients/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ł
train/gradients/zeros_1Fill'train/gradients/zeros_1/shape_as_tensortrain/gradients/zeros_1/Const*
_output_shapes
:	@*
T0*

index_type0
x
'train/gradients/zeros_2/shape_as_tensorConst*
valueB"@      *
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
Ł
train/gradients/zeros_2Fill'train/gradients/zeros_2/shape_as_tensortrain/gradients/zeros_2/Const*
T0*

index_type0*
_output_shapes
:	@
x
'train/gradients/zeros_3/shape_as_tensorConst*
valueB"@      *
dtype0*
_output_shapes
:
b
train/gradients/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ł
train/gradients/zeros_3Fill'train/gradients/zeros_3/shape_as_tensortrain/gradients/zeros_3/Const*
T0*

index_type0*
_output_shapes
:	@
­
/train/gradients/lm/rnn/while/Exit_2_grad/b_exitEnterhtrain/gradients/lm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
ă
/train/gradients/lm/rnn/while/Exit_3_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@*:

frame_name,*train/gradients/lm/rnn/while/while_context
ĺ
/train/gradients/lm/rnn/while/Exit_4_grad/b_exitEntertrain/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@*:

frame_name,*train/gradients/lm/rnn/while/while_context
ĺ
/train/gradients/lm/rnn/while/Exit_5_grad/b_exitEntertrain/gradients/zeros_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@*:

frame_name,*train/gradients/lm/rnn/while/while_context
ĺ
/train/gradients/lm/rnn/while/Exit_6_grad/b_exitEntertrain/gradients/zeros_3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	@*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ő
3train/gradients/lm/rnn/while/Switch_2_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_2_grad/b_exit:train/gradients/lm/rnn/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
Ţ
3train/gradients/lm/rnn/while/Switch_3_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_3_grad/b_exit:train/gradients/lm/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@: 
Ţ
3train/gradients/lm/rnn/while/Switch_4_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_4_grad/b_exit:train/gradients/lm/rnn/while/Switch_4_grad_1/NextIteration*
N*!
_output_shapes
:	@: *
T0
Ţ
3train/gradients/lm/rnn/while/Switch_5_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_5_grad/b_exit:train/gradients/lm/rnn/while/Switch_5_grad_1/NextIteration*
N*!
_output_shapes
:	@: *
T0
Ţ
3train/gradients/lm/rnn/while/Switch_6_grad/b_switchMerge/train/gradients/lm/rnn/while/Exit_6_grad/b_exit:train/gradients/lm/rnn/while/Switch_6_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	@: 
ő
0train/gradients/lm/rnn/while/Merge_2_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 

0train/gradients/lm/rnn/while/Merge_3_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2**
_output_shapes
:	@:	@*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_3_grad/b_switch

0train/gradients/lm/rnn/while/Merge_4_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_4_grad/b_switchtrain/gradients/b_count_2*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_4_grad/b_switch**
_output_shapes
:	@:	@*
T0

0train/gradients/lm/rnn/while/Merge_5_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_5_grad/b_switchtrain/gradients/b_count_2**
_output_shapes
:	@:	@*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_5_grad/b_switch

0train/gradients/lm/rnn/while/Merge_6_grad/SwitchSwitch3train/gradients/lm/rnn/while/Switch_6_grad/b_switchtrain/gradients/b_count_2**
_output_shapes
:	@:	@*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_6_grad/b_switch

.train/gradients/lm/rnn/while/Enter_2_grad/ExitExit0train/gradients/lm/rnn/while/Merge_2_grad/Switch*
T0*
_output_shapes
: 

.train/gradients/lm/rnn/while/Enter_3_grad/ExitExit0train/gradients/lm/rnn/while/Merge_3_grad/Switch*
_output_shapes
:	@*
T0

.train/gradients/lm/rnn/while/Enter_4_grad/ExitExit0train/gradients/lm/rnn/while/Merge_4_grad/Switch*
_output_shapes
:	@*
T0

.train/gradients/lm/rnn/while/Enter_5_grad/ExitExit0train/gradients/lm/rnn/while/Merge_5_grad/Switch*
T0*
_output_shapes
:	@

.train/gradients/lm/rnn/while/Enter_6_grad/ExitExit0train/gradients/lm/rnn/while/Merge_6_grad/Switch*
T0*
_output_shapes
:	@
Ť
gtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3mtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
sourcetrain/gradients*
_output_shapes

:: 
ę
mtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlm/rnn/TensorArray*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5*
parallel_iterations *
is_constant(*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0
˙
ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1h^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*O
_classE
CAloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5
Ě
Wtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3gtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:	@
Ô
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst**
_class 
loc:@lm/rnn/while/Identity_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
˛
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const**
_class 
loc:@lm/rnn/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
Ä
]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
´
ctrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlm/rnn/while/Identity_1^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 

btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
ß
htrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(
Ő
^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerc^train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2i^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2g^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2c^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2_^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2a^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
¤
train/gradients/AddNAddN2train/gradients/lm/rnn/while/Merge_6_grad/Switch:1Wtrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
N*
_output_shapes
:	@*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_6_grad/b_switch
ţ
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/MulMultrain/gradients/AddN`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ű
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
×
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
â
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1Multrain/gradients/AddNbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
ú
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3*
valueB :
˙˙˙˙˙˙˙˙˙
Ř
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3*

stack_name *
_output_shapes
:*
	elem_type0
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
ă
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ě
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3_grad/TanhGradTanhGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul*
_output_shapes
:	@*
T0
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1*
T0*
_output_shapes
:	@
 
:train/gradients/lm/rnn/while/Switch_2_grad_1/NextIterationNextIteration2train/gradients/lm/rnn/while/Merge_2_grad/Switch:1*
_output_shapes
: *
T0
Ş
train/gradients/AddN_1AddN2train/gradients/lm/rnn/while/Merge_5_grad/Switch:1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_3_grad/TanhGrad*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_5_grad/b_switch*
N*
_output_shapes
:	@

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/MulMultrain/gradients/AddN_1`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ű
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
×
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
â
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1Multrain/gradients/AddN_1btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
Ô
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/ConstConst**
_class 
loc:@lm/rnn/while/Identity_5*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
˛
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Const**
_class 
loc:@lm/rnn/while/Identity_5*

stack_name *
_output_shapes
:*
	elem_type0
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
˝
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Enterlm/rnn/while/Identity_5^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/MulMultrain/gradients/AddN_1`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ř
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/ConstConst*
_output_shapes
: *P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ô
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0*
is_constant(
ß
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1Multrain/gradients/AddN_1btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
ý
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ű
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ć
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1*
T0*
_output_shapes
:	@
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4_grad/SigmoidGradSigmoidGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul*
T0*
_output_shapes
:	@
Ě
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2_grad/TanhGradTanhGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1*
_output_shapes
:	@*
T0
Ě
:train/gradients/lm/rnn/while/Switch_5_grad_1/NextIterationNextIterationUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul*
_output_shapes
:	@*
T0
ž
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ShapeConst^train/gradients/Sub*
valueB"@      *
dtype0*
_output_shapes
:
˛
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
ń
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ShapeYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
č
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/SumSumatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradgtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ě
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/ReshapeReshapeUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/SumWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:	@
ě
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Sum_1Sumatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_3_grad/SigmoidGraditrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
É
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Reshape_1ReshapeWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Sum_1Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concatConcatV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_4_grad/SigmoidGrad[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_2_grad/TanhGradYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_2_grad/Reshapeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_5_grad/SigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat/Const*
_output_shapes
:	@*

Tidx0*
T0*
N
¸
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGradBiasAddGradZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat*
data_formatNHWC*
_output_shapes	
:*
T0
ô
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulMatMulZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concatatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/Enter*
T0*
_output_shapes
:	@*
transpose_a( *
transpose_b(
ś
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*:

frame_name,*train/gradients/lm/rnn/while/while_context
ţ
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1MatMulhtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_1_grad/concat*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/ConstConst*R
_classH
FDloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ć
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_accStackV2ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Const*R
_classH
FDloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1*

stack_name *
_output_shapes
:*
	elem_type0
Đ
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/EnterEnterctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ń
itrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPushV2StackPushV2ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Enter?lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
Ž
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2
StackPopV2ntrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes
:	@
ë
ntrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/StackPopV2/EnterEnterctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(
˛
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConstConst^train/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
ą
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/RankConst^train/gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
ź
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/modFloorModZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConstYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
Á
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ShapeConst^train/gradients/Sub*
valueB"@      *
dtype0*
_output_shapes
:
Ă
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1Const^train/gradients/Sub*
valueB"@      *
dtype0*
_output_shapes
:
°
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffsetConcatOffsetXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/modZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1*
N* 
_output_shapes
::
ś
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/SliceSlice[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffsetZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape*
_output_shapes
:	@*
Index0*
T0
ź
\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice_1Slice[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMulctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/ConcatOffset:1\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Shape_1*
_output_shapes
:	@*
Index0*
T0
Š
train/gradients/AddN_2AddN2train/gradients/lm/rnn/while/Merge_4_grad/Switch:1Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_4_grad/b_switch*
N*
_output_shapes
:	@

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/MulMultrain/gradients/AddN_2`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ű
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
×
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Const*
_output_shapes
:*
	elem_type0*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2*

stack_name 
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
â
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1Multrain/gradients/AddN_2btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
ú
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ř
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Const*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ă
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ě
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul*
T0*
_output_shapes
:	@
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1*
T0*
_output_shapes
:	@
Ó
:train/gradients/lm/rnn/while/Switch_6_grad_1/NextIterationNextIteration\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_1_grad/Slice_1*
T0*
_output_shapes
:	@
Ş
train/gradients/AddN_3AddN2train/gradients/lm/rnn/while/Merge_3_grad/Switch:1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
_output_shapes
:	@*
T0*F
_class<
:8loc:@train/gradients/lm/rnn/while/Switch_3_grad/b_switch*
N
ü
Strain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/MulMultrain/gradients/AddN_3^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
÷
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*
valueB :
˙˙˙˙˙˙˙˙˙
Ń
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Const*Q
_classG
ECloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
ź
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/EnterEnterYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
Ü
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Enter>lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2dtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
×
dtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1Multrain/gradients/AddN_3`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	@
Ň
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*
_output_shapes
: **
_class 
loc:@lm/rnn/while/Identity_3*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ž
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Const**
_class 
loc:@lm/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
š
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Enterlm/rnn/while/Identity_3^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/MulMultrain/gradients/AddN_3`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes
:	@
ö
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*N
_classD
B@loc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ň
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Const*
_output_shapes
:*
	elem_type0*N
_classD
B@loc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh*

stack_name 
Ŕ
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
Ý
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Enter;lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh^train/gradients/Add*
T0*
_output_shapes
:	@*
swap_memory( 

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
Ű
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnter[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 

Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1Multrain/gradients/AddN_3btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
_output_shapes
:	@*
T0
ý
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*
_output_shapes
: *S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ű
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Const*S
_classI
GEloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
Ä
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context*
T0
ć
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Enter@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
˘
btrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ß
htrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnter]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
Ď
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes
:	@*
T0
Ő
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul*
_output_shapes
:	@*
T0
Ę
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1*
T0*
_output_shapes
:	@
Ę
:train/gradients/lm/rnn/while/Switch_3_grad_1/NextIterationNextIterationStrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul*
_output_shapes
:	@*
T0
ź
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ShapeConst^train/gradients/Sub*
valueB"@      *
dtype0*
_output_shapes
:
°
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
ë
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ShapeWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
â
Strain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/SumSum_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradetrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ć
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/ReshapeReshapeStrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/SumUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	@
ć
Utrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Sum_1Sum_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradgtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ă
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Reshape_1ReshapeUtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Sum_1Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ý
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concatConcatV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Tanh_grad/TanhGradWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_grad/Reshapeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat/Const*
N*
_output_shapes
:	@*

Tidx0*
T0
ś
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:*
T0
đ
Ytrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMulXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concatatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul/Enter*
T0*
_output_shapes
:	@*
transpose_a( *
transpose_b(
ř
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMulftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/split_grad/concat*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ţ
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*
_output_shapes
: *P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ŕ
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Const*
_output_shapes
:*
	elem_type0*P
_classF
DBloc:@lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat*

stack_name 
Ě
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnteratrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:**

frame_namelm/rnn/while/while_context
ë
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter=lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat^train/gradients/Add*
_output_shapes
:	@*
swap_memory( *
T0
Ş
ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2ltrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
:	@*
	elem_type0
ç
ltrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnteratrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context

train/gradients/AddN_4AddNatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad*
N*
_output_shapes	
:*
T0
Ž
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:*
valueB*    
Ű
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:*:

frame_name,*train/gradients/lm/rnn/while/while_context
ç
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Mergeatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes
	:: *
T0

`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*"
_output_shapes
::

]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAddbtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1train/gradients/AddN_4*
T0*
_output_shapes	
:
ý
gtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
ń
atrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
°
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
Ż
Wtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
ś
Vtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/modFloorModXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConstWtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ż
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ShapeConst^train/gradients/Sub*
_output_shapes
:*
valueB"@      *
dtype0
Á
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1Const^train/gradients/Sub*
_output_shapes
:*
valueB"@      *
dtype0
¨
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetVtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/modXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ShapeZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::
Ž
Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/SliceSliceYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetXtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape*
Index0*
T0*
_output_shapes
:	@
´
Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Slice_1SliceYtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1Ztrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Shape_1*
Index0*
T0*
_output_shapes
:	@

train/gradients/AddN_5AddN]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
*
T0*p
_classf
dbloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1
ˇ
^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
*    *
dtype0* 
_output_shapes
:

Ţ
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter^train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
*:

frame_name,*train/gradients/lm/rnn/while/while_context
é
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
: 

_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*,
_output_shapes
:
:
*
T0

\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAddatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1train/gradients/AddN_5* 
_output_shapes
:
*
T0

ftrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIteration\train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
*
T0
ô
`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
*
T0
°
Utrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3[train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^train/gradients/Sub*
_output_shapes

:: *7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
sourcetrain/gradients
Â
[train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlm/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
í
]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context
î
Qtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity]train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1V^train/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*7
_class-
+)loc:@lm/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
ö
Wtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Utrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3btrain/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Xtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/SliceQtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0

Atrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
_output_shapes
: *
valueB
 *    *
dtype0

Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterAtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
is_constant( *
parallel_iterations *
_output_shapes
: *:

frame_name,*train/gradients/lm/rnn/while/while_context*
T0

Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Itrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
Ď
Btrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2train/gradients/b_count_2*
T0*
_output_shapes
: : 

?train/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddDtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Wtrain/gradients/lm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
ź
Itrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration?train/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0
°
Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitBtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0
Ń
:train/gradients/lm/rnn/while/Switch_4_grad_1/NextIterationNextIterationZtrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes
:	@*
T0
Ě
xtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lm/rnn/TensorArray_1Ctrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *'
_class
loc:@lm/rnn/TensorArray_1*
sourcetrain/gradients

ttrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityCtrain/gradients/lm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3y^train/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*'
_class
loc:@lm/rnn/TensorArray_1*
_output_shapes
: 
ä
jtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3xtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lm/rnn/TensorArrayUnstack/rangettrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@

7train/gradients/lm/rnn/transpose_grad/InvertPermutationInvertPermutationlm/rnn/concat*
_output_shapes
:*
T0
Ľ
/train/gradients/lm/rnn/transpose_grad/transpose	Transposejtrain/gradients/lm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV37train/gradients/lm/rnn/transpose_grad/InvertPermutation*,
_output_shapes
:@˙˙˙˙˙˙˙˙˙*
Tperm0*
T0
Ľ
.train/gradients/lm/embedding_lookup_grad/ShapeConst*
_output_shapes
:*
_class
loc:@embedding*%
valueB	"Ţ             *
dtype0	
ş
0train/gradients/lm/embedding_lookup_grad/ToInt32Cast.train/gradients/lm/embedding_lookup_grad/Shape*
_output_shapes
:*

DstT0*

SrcT0	*
_class
loc:@embedding
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
Ú
3train/gradients/lm/embedding_lookup_grad/ExpandDims
ExpandDims-train/gradients/lm/embedding_lookup_grad/Size7train/gradients/lm/embedding_lookup_grad/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0

<train/gradients/lm/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ě
6train/gradients/lm/embedding_lookup_grad/strided_sliceStridedSlice0train/gradients/lm/embedding_lookup_grad/ToInt32<train/gradients/lm/embedding_lookup_grad/strided_slice/stack>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_1>train/gradients/lm/embedding_lookup_grad/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0
v
4train/gradients/lm/embedding_lookup_grad/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0

/train/gradients/lm/embedding_lookup_grad/concatConcatV23train/gradients/lm/embedding_lookup_grad/ExpandDims6train/gradients/lm/embedding_lookup_grad/strided_slice4train/gradients/lm/embedding_lookup_grad/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ţ
0train/gradients/lm/embedding_lookup_grad/ReshapeReshape/train/gradients/lm/rnn/transpose_grad/transpose/train/gradients/lm/embedding_lookup_grad/concat*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
2train/gradients/lm/embedding_lookup_grad/Reshape_1Reshapeinput/Placeholder3train/gradients/lm/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
train/global_norm/L2LossL2Loss0train/gradients/lm/embedding_lookup_grad/Reshape*
_output_shapes
: *
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape

train/global_norm/L2Loss_1L2Loss`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *
T0

train/global_norm/L2Loss_2L2Lossatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *
T0*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
´
train/global_norm/L2Loss_3L2Loss,train/gradients/softmax/MatMul_grad/MatMul_1*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1*
_output_shapes
: 
ź
train/global_norm/L2Loss_4L2Loss0train/gradients/softmax/BiasAdd_grad/BiasAddGrad*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
ă
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

train/global_norm/SumSumtrain/global_norm/stacktrain/global_norm/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
#train/clip_by_global_norm/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

!train/clip_by_global_norm/truedivRealDiv#train/clip_by_global_norm/truediv/xtrain/global_norm/global_norm*
T0*
_output_shapes
: 
d
train/clip_by_global_norm/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
%train/clip_by_global_norm/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 

#train/clip_by_global_norm/truediv_1RealDivtrain/clip_by_global_norm/Const%train/clip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 

!train/clip_by_global_norm/MinimumMinimum!train/clip_by_global_norm/truediv#train/clip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
d
train/clip_by_global_norm/mul/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 

train/clip_by_global_norm/mulMultrain/clip_by_global_norm/mul/x!train/clip_by_global_norm/Minimum*
T0*
_output_shapes
: 
ď
train/clip_by_global_norm/mul_1Mul0train/gradients/lm/embedding_lookup_grad/Reshapetrain/clip_by_global_norm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape
Ű
6train/clip_by_global_norm/train/clip_by_global_norm/_0Identitytrain/clip_by_global_norm/mul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*C
_class9
75loc:@train/gradients/lm/embedding_lookup_grad/Reshape
Ç
train/clip_by_global_norm/mul_2Mul`train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3train/clip_by_global_norm/mul* 
_output_shapes
:
*
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3

6train/clip_by_global_norm/train/clip_by_global_norm/_1Identitytrain/clip_by_global_norm/mul_2* 
_output_shapes
:
*
T0*s
_classi
geloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3
Ä
train/clip_by_global_norm/mul_3Mulatrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3train/clip_by_global_norm/mul*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:*
T0
˙
6train/clip_by_global_norm/train/clip_by_global_norm/_2Identitytrain/clip_by_global_norm/mul_3*t
_classj
hfloc:@train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes	
:*
T0
ß
train/clip_by_global_norm/mul_4Mul,train/gradients/softmax/MatMul_grad/MatMul_1train/clip_by_global_norm/mul* 
_output_shapes
:
Ţ/*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1
Ď
6train/clip_by_global_norm/train/clip_by_global_norm/_3Identitytrain/clip_by_global_norm/mul_4*
T0*?
_class5
31loc:@train/gradients/softmax/MatMul_grad/MatMul_1* 
_output_shapes
:
Ţ/
â
train/clip_by_global_norm/mul_5Mul0train/gradients/softmax/BiasAdd_grad/BiasAddGradtrain/clip_by_global_norm/mul*
_output_shapes	
:Ţ/*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad
Î
6train/clip_by_global_norm/train/clip_by_global_norm/_4Identitytrain/clip_by_global_norm/mul_5*
_output_shapes	
:Ţ/*
T0*C
_class9
75loc:@train/gradients/softmax/BiasAdd_grad/BiasAddGrad

train/beta1_power/initial_valueConst*
_class
loc:@embedding*
valueB
 *fff?*
dtype0*
_output_shapes
: 

train/beta1_power
VariableV2*
shared_name *
_class
loc:@embedding*
	container *
shape: *
dtype0*
_output_shapes
: 
ž
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(
t
train/beta1_power/readIdentitytrain/beta1_power*
T0*
_class
loc:@embedding*
_output_shapes
: 

train/beta2_power/initial_valueConst*
_class
loc:@embedding*
valueB
 *wž?*
dtype0*
_output_shapes
: 

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
ž
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
t
train/beta2_power/readIdentitytrain/beta2_power*
_output_shapes
: *
T0*
_class
loc:@embedding

0embedding/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@embedding*
valueB"Ţ     *
dtype0*
_output_shapes
:

&embedding/Adam/Initializer/zeros/ConstConst*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
Ý
 embedding/Adam/Initializer/zerosFill0embedding/Adam/Initializer/zeros/shape_as_tensor&embedding/Adam/Initializer/zeros/Const*
T0*
_class
loc:@embedding*

index_type0* 
_output_shapes
:
Ţ/
¤
embedding/Adam
VariableV2*
shape:
Ţ/*
dtype0* 
_output_shapes
:
Ţ/*
shared_name *
_class
loc:@embedding*
	container 
Ă
embedding/Adam/AssignAssignembedding/Adam embedding/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/
x
embedding/Adam/readIdentityembedding/Adam*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Ą
2embedding/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@embedding*
valueB"Ţ     *
dtype0*
_output_shapes
:

(embedding/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
ă
"embedding/Adam_1/Initializer/zerosFill2embedding/Adam_1/Initializer/zeros/shape_as_tensor(embedding/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@embedding*

index_type0* 
_output_shapes
:
Ţ/
Ś
embedding/Adam_1
VariableV2*
_class
loc:@embedding*
	container *
shape:
Ţ/*
dtype0* 
_output_shapes
:
Ţ/*
shared_name 
É
embedding/Adam_1/AssignAssignembedding/Adam_1"embedding/Adam_1/Initializer/zeros*
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/*
use_locking(
|
embedding/Adam_1/readIdentityembedding/Adam_1* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@embedding
í
Wrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
×
Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *    
ů
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zerosFillWrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorMrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*

index_type0
ň
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ß
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
í
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ď
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
˙
Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*

index_type0* 
_output_shapes
:

ô
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape:

ĺ
>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:

ń
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1* 
_output_shapes
:
*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
×
Ernn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB*    *
dtype0*
_output_shapes	
:
ä
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:
Ň
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:
â
8rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:*
T0
Ů
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB*    
ć
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:
Ř
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:
ć
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

0softmax_w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_w*
valueB"   Ţ  *
dtype0*
_output_shapes
:

&softmax_w/Adam/Initializer/zeros/ConstConst*
_class
loc:@softmax_w*
valueB
 *    *
dtype0*
_output_shapes
: 
Ý
 softmax_w/Adam/Initializer/zerosFill0softmax_w/Adam/Initializer/zeros/shape_as_tensor&softmax_w/Adam/Initializer/zeros/Const* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@softmax_w*

index_type0
¤
softmax_w/Adam
VariableV2*
dtype0* 
_output_shapes
:
Ţ/*
shared_name *
_class
loc:@softmax_w*
	container *
shape:
Ţ/
Ă
softmax_w/Adam/AssignAssignsoftmax_w/Adam softmax_w/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@softmax_w*
validate_shape(* 
_output_shapes
:
Ţ/
x
softmax_w/Adam/readIdentitysoftmax_w/Adam*
_class
loc:@softmax_w* 
_output_shapes
:
Ţ/*
T0
Ą
2softmax_w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_w*
valueB"   Ţ  *
dtype0*
_output_shapes
:

(softmax_w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@softmax_w*
valueB
 *    *
dtype0*
_output_shapes
: 
ă
"softmax_w/Adam_1/Initializer/zerosFill2softmax_w/Adam_1/Initializer/zeros/shape_as_tensor(softmax_w/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@softmax_w*

index_type0
Ś
softmax_w/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
Ţ/*
shared_name *
_class
loc:@softmax_w*
	container *
shape:
Ţ/
É
softmax_w/Adam_1/AssignAssignsoftmax_w/Adam_1"softmax_w/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@softmax_w*
validate_shape(* 
_output_shapes
:
Ţ/
|
softmax_w/Adam_1/readIdentitysoftmax_w/Adam_1* 
_output_shapes
:
Ţ/*
T0*
_class
loc:@softmax_w

0softmax_b/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@softmax_b*
valueB:Ţ/*
dtype0*
_output_shapes
:

&softmax_b/Adam/Initializer/zeros/ConstConst*
_class
loc:@softmax_b*
valueB
 *    *
dtype0*
_output_shapes
: 
Ř
 softmax_b/Adam/Initializer/zerosFill0softmax_b/Adam/Initializer/zeros/shape_as_tensor&softmax_b/Adam/Initializer/zeros/Const*
_class
loc:@softmax_b*

index_type0*
_output_shapes	
:Ţ/*
T0

softmax_b/Adam
VariableV2*
_class
loc:@softmax_b*
	container *
shape:Ţ/*
dtype0*
_output_shapes	
:Ţ/*
shared_name 
ž
softmax_b/Adam/AssignAssignsoftmax_b/Adam softmax_b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:Ţ/*
use_locking(*
T0*
_class
loc:@softmax_b
s
softmax_b/Adam/readIdentitysoftmax_b/Adam*
T0*
_class
loc:@softmax_b*
_output_shapes	
:Ţ/

2softmax_b/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@softmax_b*
valueB:Ţ/*
dtype0

(softmax_b/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@softmax_b*
valueB
 *    *
dtype0
Ţ
"softmax_b/Adam_1/Initializer/zerosFill2softmax_b/Adam_1/Initializer/zeros/shape_as_tensor(softmax_b/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@softmax_b*

index_type0*
_output_shapes	
:Ţ/

softmax_b/Adam_1
VariableV2*
dtype0*
_output_shapes	
:Ţ/*
shared_name *
_class
loc:@softmax_b*
	container *
shape:Ţ/
Ä
softmax_b/Adam_1/AssignAssignsoftmax_b/Adam_1"softmax_b/Adam_1/Initializer/zeros*
_output_shapes	
:Ţ/*
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
:Ţ/
U
train/Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
U
train/Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ę
"train/Adam/update_embedding/UniqueUnique2train/gradients/lm/embedding_lookup_grad/Reshape_1*
out_idx0*
T0*
_class
loc:@embedding*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ą
!train/Adam/update_embedding/ShapeShape"train/Adam/update_embedding/Unique*
T0*
_class
loc:@embedding*
out_type0*
_output_shapes
:

/train/Adam/update_embedding/strided_slice/stackConst*
_class
loc:@embedding*
valueB: *
dtype0*
_output_shapes
:

1train/Adam/update_embedding/strided_slice/stack_1Const*
_class
loc:@embedding*
valueB:*
dtype0*
_output_shapes
:

1train/Adam/update_embedding/strided_slice/stack_2Const*
_output_shapes
:*
_class
loc:@embedding*
valueB:*
dtype0
Ł
)train/Adam/update_embedding/strided_sliceStridedSlice!train/Adam/update_embedding/Shape/train/Adam/update_embedding/strided_slice/stack1train/Adam/update_embedding/strided_slice/stack_11train/Adam/update_embedding/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
_class
loc:@embedding
Â
.train/Adam/update_embedding/UnsortedSegmentSumUnsortedSegmentSum6train/clip_by_global_norm/train/clip_by_global_norm/_0$train/Adam/update_embedding/Unique:1)train/Adam/update_embedding/strided_slice*
T0*
_class
loc:@embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tnumsegments0*
Tindices0

!train/Adam/update_embedding/sub/xConst*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
 
train/Adam/update_embedding/subSub!train/Adam/update_embedding/sub/xtrain/beta2_power/read*
T0*
_class
loc:@embedding*
_output_shapes
: 

 train/Adam/update_embedding/SqrtSqrttrain/Adam/update_embedding/sub*
T0*
_class
loc:@embedding*
_output_shapes
: 
¤
train/Adam/update_embedding/mulMullearning_rate/Variable/read train/Adam/update_embedding/Sqrt*
_output_shapes
: *
T0*
_class
loc:@embedding

#train/Adam/update_embedding/sub_1/xConst*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *  ?*
dtype0
¤
!train/Adam/update_embedding/sub_1Sub#train/Adam/update_embedding/sub_1/xtrain/beta1_power/read*
_output_shapes
: *
T0*
_class
loc:@embedding
ą
#train/Adam/update_embedding/truedivRealDivtrain/Adam/update_embedding/mul!train/Adam/update_embedding/sub_1*
_class
loc:@embedding*
_output_shapes
: *
T0

#train/Adam/update_embedding/sub_2/xConst*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 

!train/Adam/update_embedding/sub_2Sub#train/Adam/update_embedding/sub_2/xtrain/Adam/beta1*
T0*
_class
loc:@embedding*
_output_shapes
: 
Ě
!train/Adam/update_embedding/mul_1Mul.train/Adam/update_embedding/UnsortedSegmentSum!train/Adam/update_embedding/sub_2*
T0*
_class
loc:@embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

!train/Adam/update_embedding/mul_2Mulembedding/Adam/readtrain/Adam/beta1*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Ń
"train/Adam/update_embedding/AssignAssignembedding/Adam!train/Adam/update_embedding/mul_2*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/*
use_locking( *
T0

&train/Adam/update_embedding/ScatterAdd
ScatterAddembedding/Adam"train/Adam/update_embedding/Unique!train/Adam/update_embedding/mul_1#^train/Adam/update_embedding/Assign*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Ů
!train/Adam/update_embedding/mul_3Mul.train/Adam/update_embedding/UnsortedSegmentSum.train/Adam/update_embedding/UnsortedSegmentSum*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@embedding

#train/Adam/update_embedding/sub_3/xConst*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 

!train/Adam/update_embedding/sub_3Sub#train/Adam/update_embedding/sub_3/xtrain/Adam/beta2*
T0*
_class
loc:@embedding*
_output_shapes
: 
ż
!train/Adam/update_embedding/mul_4Mul!train/Adam/update_embedding/mul_3!train/Adam/update_embedding/sub_3*
T0*
_class
loc:@embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

!train/Adam/update_embedding/mul_5Mulembedding/Adam_1/readtrain/Adam/beta2*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Ő
$train/Adam/update_embedding/Assign_1Assignembedding/Adam_1!train/Adam/update_embedding/mul_5*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(* 
_output_shapes
:
Ţ/
˘
(train/Adam/update_embedding/ScatterAdd_1
ScatterAddembedding/Adam_1"train/Adam/update_embedding/Unique!train/Adam/update_embedding/mul_4%^train/Adam/update_embedding/Assign_1*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/*
use_locking( *
Tindices0

"train/Adam/update_embedding/Sqrt_1Sqrt(train/Adam/update_embedding/ScatterAdd_1*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
ž
!train/Adam/update_embedding/mul_6Mul#train/Adam/update_embedding/truediv&train/Adam/update_embedding/ScatterAdd*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
§
train/Adam/update_embedding/addAdd"train/Adam/update_embedding/Sqrt_1train/Adam/epsilon*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
˝
%train/Adam/update_embedding/truediv_1RealDiv!train/Adam/update_embedding/mul_6train/Adam/update_embedding/add*
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Ŕ
%train/Adam/update_embedding/AssignSub	AssignSub	embedding%train/Adam/update_embedding/truediv_1*
use_locking( *
T0*
_class
loc:@embedding* 
_output_shapes
:
Ţ/
Č
&train/Adam/update_embedding/group_depsNoOp&^train/Adam/update_embedding/AssignSub'^train/Adam/update_embedding/ScatterAdd)^train/Adam/update_embedding/ScatterAdd_1*
_class
loc:@embedding
Ď
Ltrain/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_1* 
_output_shapes
:
*
use_locking( *
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
use_nesterov( 
Ŕ
Jtrain/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_2*
T0*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 

%train/Adam/update_softmax_w/ApplyAdam	ApplyAdam	softmax_wsoftmax_w/Adamsoftmax_w/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_3*
_class
loc:@softmax_w*
use_nesterov( * 
_output_shapes
:
Ţ/*
use_locking( *
T0

%train/Adam/update_softmax_b/ApplyAdam	ApplyAdam	softmax_bsoftmax_b/Adamsoftmax_b/Adam_1train/beta1_power/readtrain/beta2_power/readlearning_rate/Variable/readtrain/Adam/beta1train/Adam/beta2train/Adam/epsilon6train/clip_by_global_norm/train/clip_by_global_norm/_4*
use_nesterov( *
_output_shapes	
:Ţ/*
use_locking( *
T0*
_class
loc:@softmax_b

train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@embedding
Ś
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 

train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@embedding
Ş
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*
_output_shapes
: 
Ń

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1'^train/Adam/update_embedding/group_depsK^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamM^train/Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam&^train/Adam/update_softmax_b/ApplyAdam&^train/Adam/update_softmax_w/ApplyAdam

initNoOp^embedding/Adam/Assign^embedding/Adam_1/Assign^embedding/Assign^learning_rate/Variable/Assign;^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign6^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign8^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign^softmax_b/Adam/Assign^softmax_b/Adam_1/Assign^softmax_b/Assign^softmax_w/Adam/Assign^softmax_w/Adam_1/Assign^softmax_w/Assign^train/beta1_power/Assign^train/beta2_power/Assign
\
Merge/MergeSummaryMergeSummarylearning_rate_1loss_1*
N*
_output_shapes
: ""
train_op


train/Adam"l
while_contextll
l
lm/rnn/while/while_context *lm/rnn/while/LoopCond:02lm/rnn/while/Merge:0:lm/rnn/while/Identity:0Blm/rnn/while/Exit:0Blm/rnn/while/Exit_1:0Blm/rnn/while/Exit_2:0Blm/rnn/while/Exit_3:0Blm/rnn/while/Exit_4:0Blm/rnn/while/Exit_5:0Blm/rnn/while/Exit_6:0Btrain/gradients/f_count_2:0Jh
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
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0ş
[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/f_acc:0[train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul/Enter:0ž
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul/Enter:03
lm/rnn/strided_slice:0lm/rnn/while/Less/Enter:0ž
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul/Enter:0
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0Elm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul/Enter:0
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0Flm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/BiasAdd/Enter:0ž
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul/Enter:0Î
etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/f_acc:0etrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_1_grad/MatMul_1/Enter:0/
lm/rnn/Minimum:0lm/rnn/while/Less_1/Enter:0Â
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul_1/Enter:0Â
_train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0_train/gradients/lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0@
lm/rnn/TensorArray_1:0&lm/rnn/while/TensorArrayReadV3/Enter:0Â
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_4_grad/Mul_1/Enter:0Ę
ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0ctrain/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0ž
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_5_grad/Mul/Enter:0Â
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_3_grad/Mul_1/Enter:0Â
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0o
Clm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(lm/rnn/while/TensorArrayReadV3/Enter_1:0Â
_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0_train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0P
lm/rnn/TensorArray:08lm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0ž
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_1_grad/Mul/Enter:0ž
]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0]train/gradients/lm/rnn/while/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_grad/Mul_1/Enter:0Rlm/rnn/while/Enter:0Rlm/rnn/while/Enter_1:0Rlm/rnn/while/Enter_2:0Rlm/rnn/while/Enter_3:0Rlm/rnn/while/Enter_4:0Rlm/rnn/while/Enter_5:0Rlm/rnn/while/Enter_6:0Rtrain/gradients/f_count_1:0Zlm/rnn/strided_slice:0"č
	variablesÚ×

learning_rate/Variable:0learning_rate/Variable/Assignlearning_rate/Variable/read:02&learning_rate/Variable/initial_value:0
Y
embedding:0embedding/Assignembedding/read:02&embedding/Initializer/random_uniform:0
ő
2rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
ä
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

7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam:0<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/read:02Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros:0

9rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1:0>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/read:02Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
ř
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam:0:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/read:02Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros:0

7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1:0<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/read:02Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
d
softmax_w/Adam:0softmax_w/Adam/Assignsoftmax_w/Adam/read:02"softmax_w/Adam/Initializer/zeros:0
l
softmax_w/Adam_1:0softmax_w/Adam_1/Assignsoftmax_w/Adam_1/read:02$softmax_w/Adam_1/Initializer/zeros:0
d
softmax_b/Adam:0softmax_b/Adam/Assignsoftmax_b/Adam/read:02"softmax_b/Adam/Initializer/zeros:0
l
softmax_b/Adam_1:0softmax_b/Adam_1/Assignsoftmax_b/Adam_1/read:02$softmax_b/Adam_1/Initializer/zeros:0"
trainable_variablesóđ
Y
embedding:0embedding/Assignembedding/read:02&embedding/Initializer/random_uniform:0
ő
2rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
ä
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/zeros:0
Y
softmax_w:0softmax_w/Assignsoftmax_w/read:02&softmax_w/Initializer/random_uniform:0
Y
softmax_b:0softmax_b/Assignsoftmax_b/read:02&softmax_b/Initializer/random_uniform:0",
	summaries

learning_rate_1:0
loss_1:0đ˛62       $Vě	´ľ¤:ŢĘÖA*'

learning_rate_1o;

loss_1=Aí6