ЏИ
н–
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ъ
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

њ
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
ц
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И"train*1.14.02unknown8і∞
~
PlaceholderPlaceholder*$
shape:€€€€€€€€€@@*
dtype0*/
_output_shapes
:€€€€€€€€€@@
±
2encoder/e1/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@encoder/e1/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Ы
0encoder/e1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e1/kernel*
valueB
 *чь”љ*
dtype0*
_output_shapes
: 
Ы
0encoder/e1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@encoder/e1/kernel*
valueB
 *чь”=*
dtype0
г
:encoder/e1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e1/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@encoder/e1/kernel*
dtype0*&
_output_shapes
: 
в
0encoder/e1/kernel/Initializer/random_uniform/subSub0encoder/e1/kernel/Initializer/random_uniform/max0encoder/e1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e1/kernel*
_output_shapes
: 
ь
0encoder/e1/kernel/Initializer/random_uniform/mulMul:encoder/e1/kernel/Initializer/random_uniform/RandomUniform0encoder/e1/kernel/Initializer/random_uniform/sub*$
_class
loc:@encoder/e1/kernel*&
_output_shapes
: *
T0
о
,encoder/e1/kernel/Initializer/random_uniformAdd0encoder/e1/kernel/Initializer/random_uniform/mul0encoder/e1/kernel/Initializer/random_uniform/min*&
_output_shapes
: *
T0*$
_class
loc:@encoder/e1/kernel
ђ
encoder/e1/kernelVarHandleOp*
_output_shapes
: *
shape: *"
shared_nameencoder/e1/kernel*$
_class
loc:@encoder/e1/kernel*
dtype0
s
2encoder/e1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e1/kernel*
_output_shapes
: 
†
encoder/e1/kernel/AssignAssignVariableOpencoder/e1/kernel,encoder/e1/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e1/kernel*
dtype0
•
%encoder/e1/kernel/Read/ReadVariableOpReadVariableOpencoder/e1/kernel*$
_class
loc:@encoder/e1/kernel*
dtype0*&
_output_shapes
: 
Т
!encoder/e1/bias/Initializer/zerosConst*"
_class
loc:@encoder/e1/bias*
valueB *    *
dtype0*
_output_shapes
: 
Ъ
encoder/e1/biasVarHandleOp*
shape: * 
shared_nameencoder/e1/bias*"
_class
loc:@encoder/e1/bias*
dtype0*
_output_shapes
: 
o
0encoder/e1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e1/bias*
_output_shapes
: 
П
encoder/e1/bias/AssignAssignVariableOpencoder/e1/bias!encoder/e1/bias/Initializer/zeros*"
_class
loc:@encoder/e1/bias*
dtype0
У
#encoder/e1/bias/Read/ReadVariableOpReadVariableOpencoder/e1/bias*"
_class
loc:@encoder/e1/bias*
dtype0*
_output_shapes
: 
i
encoder/e1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
z
 encoder/e1/Conv2D/ReadVariableOpReadVariableOpencoder/e1/kernel*
dtype0*&
_output_shapes
: 
ђ
encoder/e1/Conv2DConv2DPlaceholder encoder/e1/Conv2D/ReadVariableOp*
strides
*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
T0
m
!encoder/e1/BiasAdd/ReadVariableOpReadVariableOpencoder/e1/bias*
_output_shapes
: *
dtype0
Н
encoder/e1/BiasAddBiasAddencoder/e1/Conv2D!encoder/e1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   
e
encoder/e1/ReluReluencoder/e1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€   
±
2encoder/e2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*$
_class
loc:@encoder/e2/kernel*%
valueB"              
Ы
0encoder/e2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e2/kernel*
valueB
 *qƒЬљ*
dtype0*
_output_shapes
: 
Ы
0encoder/e2/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@encoder/e2/kernel*
valueB
 *qƒЬ=*
dtype0*
_output_shapes
: 
г
:encoder/e2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e2/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@encoder/e2/kernel*
dtype0*&
_output_shapes
:  
в
0encoder/e2/kernel/Initializer/random_uniform/subSub0encoder/e2/kernel/Initializer/random_uniform/max0encoder/e2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e2/kernel*
_output_shapes
: 
ь
0encoder/e2/kernel/Initializer/random_uniform/mulMul:encoder/e2/kernel/Initializer/random_uniform/RandomUniform0encoder/e2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@encoder/e2/kernel*&
_output_shapes
:  
о
,encoder/e2/kernel/Initializer/random_uniformAdd0encoder/e2/kernel/Initializer/random_uniform/mul0encoder/e2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e2/kernel*&
_output_shapes
:  
ђ
encoder/e2/kernelVarHandleOp*
shape:  *"
shared_nameencoder/e2/kernel*$
_class
loc:@encoder/e2/kernel*
dtype0*
_output_shapes
: 
s
2encoder/e2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e2/kernel*
_output_shapes
: 
†
encoder/e2/kernel/AssignAssignVariableOpencoder/e2/kernel,encoder/e2/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e2/kernel*
dtype0
•
%encoder/e2/kernel/Read/ReadVariableOpReadVariableOpencoder/e2/kernel*$
_class
loc:@encoder/e2/kernel*
dtype0*&
_output_shapes
:  
Т
!encoder/e2/bias/Initializer/zerosConst*"
_class
loc:@encoder/e2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Ъ
encoder/e2/biasVarHandleOp* 
shared_nameencoder/e2/bias*"
_class
loc:@encoder/e2/bias*
dtype0*
_output_shapes
: *
shape: 
o
0encoder/e2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e2/bias*
_output_shapes
: 
П
encoder/e2/bias/AssignAssignVariableOpencoder/e2/bias!encoder/e2/bias/Initializer/zeros*
dtype0*"
_class
loc:@encoder/e2/bias
У
#encoder/e2/bias/Read/ReadVariableOpReadVariableOpencoder/e2/bias*"
_class
loc:@encoder/e2/bias*
dtype0*
_output_shapes
: 
i
encoder/e2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
z
 encoder/e2/Conv2D/ReadVariableOpReadVariableOpencoder/e2/kernel*
dtype0*&
_output_shapes
:  
∞
encoder/e2/Conv2DConv2Dencoder/e1/Relu encoder/e2/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€ 
m
!encoder/e2/BiasAdd/ReadVariableOpReadVariableOpencoder/e2/bias*
dtype0*
_output_shapes
: 
Н
encoder/e2/BiasAddBiasAddencoder/e2/Conv2D!encoder/e2/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€ *
T0
e
encoder/e2/ReluReluencoder/e2/BiasAdd*/
_output_shapes
:€€€€€€€€€ *
T0
±
2encoder/e3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*$
_class
loc:@encoder/e3/kernel*%
valueB"          @   
Ы
0encoder/e3/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e3/kernel*
valueB
 *   Њ*
dtype0*
_output_shapes
: 
Ы
0encoder/e3/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@encoder/e3/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
г
:encoder/e3/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: @*
T0*$
_class
loc:@encoder/e3/kernel
в
0encoder/e3/kernel/Initializer/random_uniform/subSub0encoder/e3/kernel/Initializer/random_uniform/max0encoder/e3/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e3/kernel*
_output_shapes
: 
ь
0encoder/e3/kernel/Initializer/random_uniform/mulMul:encoder/e3/kernel/Initializer/random_uniform/RandomUniform0encoder/e3/kernel/Initializer/random_uniform/sub*$
_class
loc:@encoder/e3/kernel*&
_output_shapes
: @*
T0
о
,encoder/e3/kernel/Initializer/random_uniformAdd0encoder/e3/kernel/Initializer/random_uniform/mul0encoder/e3/kernel/Initializer/random_uniform/min*&
_output_shapes
: @*
T0*$
_class
loc:@encoder/e3/kernel
ђ
encoder/e3/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*"
shared_nameencoder/e3/kernel*$
_class
loc:@encoder/e3/kernel
s
2encoder/e3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e3/kernel*
_output_shapes
: 
†
encoder/e3/kernel/AssignAssignVariableOpencoder/e3/kernel,encoder/e3/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e3/kernel*
dtype0
•
%encoder/e3/kernel/Read/ReadVariableOpReadVariableOpencoder/e3/kernel*$
_class
loc:@encoder/e3/kernel*
dtype0*&
_output_shapes
: @
Т
!encoder/e3/bias/Initializer/zerosConst*"
_class
loc:@encoder/e3/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ъ
encoder/e3/biasVarHandleOp*"
_class
loc:@encoder/e3/bias*
dtype0*
_output_shapes
: *
shape:@* 
shared_nameencoder/e3/bias
o
0encoder/e3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e3/bias*
_output_shapes
: 
П
encoder/e3/bias/AssignAssignVariableOpencoder/e3/bias!encoder/e3/bias/Initializer/zeros*"
_class
loc:@encoder/e3/bias*
dtype0
У
#encoder/e3/bias/Read/ReadVariableOpReadVariableOpencoder/e3/bias*"
_class
loc:@encoder/e3/bias*
dtype0*
_output_shapes
:@
i
encoder/e3/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
z
 encoder/e3/Conv2D/ReadVariableOpReadVariableOpencoder/e3/kernel*
dtype0*&
_output_shapes
: @
∞
encoder/e3/Conv2DConv2Dencoder/e2/Relu encoder/e3/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€@
m
!encoder/e3/BiasAdd/ReadVariableOpReadVariableOpencoder/e3/bias*
dtype0*
_output_shapes
:@
Н
encoder/e3/BiasAddBiasAddencoder/e3/Conv2D!encoder/e3/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@
e
encoder/e3/ReluReluencoder/e3/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
±
2encoder/e4/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@encoder/e4/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
Ы
0encoder/e4/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e4/kernel*
valueB
 *„≥Ёљ*
dtype0*
_output_shapes
: 
Ы
0encoder/e4/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@encoder/e4/kernel*
valueB
 *„≥Ё=*
dtype0
г
:encoder/e4/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e4/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*
T0*$
_class
loc:@encoder/e4/kernel
в
0encoder/e4/kernel/Initializer/random_uniform/subSub0encoder/e4/kernel/Initializer/random_uniform/max0encoder/e4/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e4/kernel*
_output_shapes
: 
ь
0encoder/e4/kernel/Initializer/random_uniform/mulMul:encoder/e4/kernel/Initializer/random_uniform/RandomUniform0encoder/e4/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@encoder/e4/kernel*&
_output_shapes
:@@
о
,encoder/e4/kernel/Initializer/random_uniformAdd0encoder/e4/kernel/Initializer/random_uniform/mul0encoder/e4/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e4/kernel*&
_output_shapes
:@@
ђ
encoder/e4/kernelVarHandleOp*
_output_shapes
: *
shape:@@*"
shared_nameencoder/e4/kernel*$
_class
loc:@encoder/e4/kernel*
dtype0
s
2encoder/e4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e4/kernel*
_output_shapes
: 
†
encoder/e4/kernel/AssignAssignVariableOpencoder/e4/kernel,encoder/e4/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e4/kernel*
dtype0
•
%encoder/e4/kernel/Read/ReadVariableOpReadVariableOpencoder/e4/kernel*&
_output_shapes
:@@*$
_class
loc:@encoder/e4/kernel*
dtype0
Т
!encoder/e4/bias/Initializer/zerosConst*"
_class
loc:@encoder/e4/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ъ
encoder/e4/biasVarHandleOp* 
shared_nameencoder/e4/bias*"
_class
loc:@encoder/e4/bias*
dtype0*
_output_shapes
: *
shape:@
o
0encoder/e4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e4/bias*
_output_shapes
: 
П
encoder/e4/bias/AssignAssignVariableOpencoder/e4/bias!encoder/e4/bias/Initializer/zeros*
dtype0*"
_class
loc:@encoder/e4/bias
У
#encoder/e4/bias/Read/ReadVariableOpReadVariableOpencoder/e4/bias*"
_class
loc:@encoder/e4/bias*
dtype0*
_output_shapes
:@
i
encoder/e4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
z
 encoder/e4/Conv2D/ReadVariableOpReadVariableOpencoder/e4/kernel*
dtype0*&
_output_shapes
:@@
∞
encoder/e4/Conv2DConv2Dencoder/e3/Relu encoder/e4/Conv2D/ReadVariableOp*
strides
*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
T0
m
!encoder/e4/BiasAdd/ReadVariableOpReadVariableOpencoder/e4/bias*
dtype0*
_output_shapes
:@
Н
encoder/e4/BiasAddBiasAddencoder/e4/Conv2D!encoder/e4/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€@*
T0
e
encoder/e4/ReluReluencoder/e4/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
T
encoder/flatten/ShapeShapeencoder/e4/Relu*
_output_shapes
:*
T0
m
#encoder/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%encoder/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%encoder/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
э
encoder/flatten/strided_sliceStridedSliceencoder/flatten/Shape#encoder/flatten/strided_slice/stack%encoder/flatten/strided_slice/stack_1%encoder/flatten/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
j
encoder/flatten/Reshape/shape/1Const*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
У
encoder/flatten/Reshape/shapePackencoder/flatten/strided_sliceencoder/flatten/Reshape/shape/1*
_output_shapes
:*
T0*
N
Е
encoder/flatten/ReshapeReshapeencoder/e4/Reluencoder/flatten/Reshape/shape*
T0*(
_output_shapes
:€€€€€€€€€А
©
2encoder/e5/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@encoder/e5/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ы
0encoder/e5/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e5/kernel*
valueB
 *М7Мљ*
dtype0*
_output_shapes
: 
Ы
0encoder/e5/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@encoder/e5/kernel*
valueB
 *М7М=*
dtype0*
_output_shapes
: 
Ё
:encoder/e5/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e5/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@encoder/e5/kernel*
dtype0* 
_output_shapes
:
АА
в
0encoder/e5/kernel/Initializer/random_uniform/subSub0encoder/e5/kernel/Initializer/random_uniform/max0encoder/e5/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e5/kernel*
_output_shapes
: 
ц
0encoder/e5/kernel/Initializer/random_uniform/mulMul:encoder/e5/kernel/Initializer/random_uniform/RandomUniform0encoder/e5/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@encoder/e5/kernel* 
_output_shapes
:
АА
и
,encoder/e5/kernel/Initializer/random_uniformAdd0encoder/e5/kernel/Initializer/random_uniform/mul0encoder/e5/kernel/Initializer/random_uniform/min*$
_class
loc:@encoder/e5/kernel* 
_output_shapes
:
АА*
T0
¶
encoder/e5/kernelVarHandleOp*
shape:
АА*"
shared_nameencoder/e5/kernel*$
_class
loc:@encoder/e5/kernel*
dtype0*
_output_shapes
: 
s
2encoder/e5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e5/kernel*
_output_shapes
: 
†
encoder/e5/kernel/AssignAssignVariableOpencoder/e5/kernel,encoder/e5/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e5/kernel*
dtype0
Я
%encoder/e5/kernel/Read/ReadVariableOpReadVariableOpencoder/e5/kernel*$
_class
loc:@encoder/e5/kernel*
dtype0* 
_output_shapes
:
АА
Ф
!encoder/e5/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*"
_class
loc:@encoder/e5/bias*
valueBА*    
Ы
encoder/e5/biasVarHandleOp*
shape:А* 
shared_nameencoder/e5/bias*"
_class
loc:@encoder/e5/bias*
dtype0*
_output_shapes
: 
o
0encoder/e5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e5/bias*
_output_shapes
: 
П
encoder/e5/bias/AssignAssignVariableOpencoder/e5/bias!encoder/e5/bias/Initializer/zeros*"
_class
loc:@encoder/e5/bias*
dtype0
Ф
#encoder/e5/bias/Read/ReadVariableOpReadVariableOpencoder/e5/bias*
dtype0*
_output_shapes	
:А*"
_class
loc:@encoder/e5/bias
t
 encoder/e5/MatMul/ReadVariableOpReadVariableOpencoder/e5/kernel*
dtype0* 
_output_shapes
:
АА
Й
encoder/e5/MatMulMatMulencoder/flatten/Reshape encoder/e5/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
n
!encoder/e5/BiasAdd/ReadVariableOpReadVariableOpencoder/e5/bias*
dtype0*
_output_shapes	
:А
Ж
encoder/e5/BiasAddBiasAddencoder/e5/MatMul!encoder/e5/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
^
encoder/e5/ReluReluencoder/e5/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
ѓ
5encoder/means/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*'
_class
loc:@encoder/means/kernel*
valueB"   
   *
dtype0
°
3encoder/means/kernel/Initializer/random_uniform/minConst*'
_class
loc:@encoder/means/kernel*
valueB
 *Ў Њ*
dtype0*
_output_shapes
: 
°
3encoder/means/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@encoder/means/kernel*
valueB
 *Ў >*
dtype0*
_output_shapes
: 
е
=encoder/means/kernel/Initializer/random_uniform/RandomUniformRandomUniform5encoder/means/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@encoder/means/kernel*
dtype0*
_output_shapes
:	А

о
3encoder/means/kernel/Initializer/random_uniform/subSub3encoder/means/kernel/Initializer/random_uniform/max3encoder/means/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@encoder/means/kernel
Б
3encoder/means/kernel/Initializer/random_uniform/mulMul=encoder/means/kernel/Initializer/random_uniform/RandomUniform3encoder/means/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@encoder/means/kernel*
_output_shapes
:	А

у
/encoder/means/kernel/Initializer/random_uniformAdd3encoder/means/kernel/Initializer/random_uniform/mul3encoder/means/kernel/Initializer/random_uniform/min*
_output_shapes
:	А
*
T0*'
_class
loc:@encoder/means/kernel
Ѓ
encoder/means/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А
*%
shared_nameencoder/means/kernel*'
_class
loc:@encoder/means/kernel
y
5encoder/means/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/means/kernel*
_output_shapes
: 
ђ
encoder/means/kernel/AssignAssignVariableOpencoder/means/kernel/encoder/means/kernel/Initializer/random_uniform*'
_class
loc:@encoder/means/kernel*
dtype0
І
(encoder/means/kernel/Read/ReadVariableOpReadVariableOpencoder/means/kernel*'
_class
loc:@encoder/means/kernel*
dtype0*
_output_shapes
:	А

Ш
$encoder/means/bias/Initializer/zerosConst*%
_class
loc:@encoder/means/bias*
valueB
*    *
dtype0*
_output_shapes
:

£
encoder/means/biasVarHandleOp*#
shared_nameencoder/means/bias*%
_class
loc:@encoder/means/bias*
dtype0*
_output_shapes
: *
shape:

u
3encoder/means/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/means/bias*
_output_shapes
: 
Ы
encoder/means/bias/AssignAssignVariableOpencoder/means/bias$encoder/means/bias/Initializer/zeros*%
_class
loc:@encoder/means/bias*
dtype0
Ь
&encoder/means/bias/Read/ReadVariableOpReadVariableOpencoder/means/bias*%
_class
loc:@encoder/means/bias*
dtype0*
_output_shapes
:

y
#encoder/means/MatMul/ReadVariableOpReadVariableOpencoder/means/kernel*
_output_shapes
:	А
*
dtype0
Ж
encoder/means/MatMulMatMulencoder/e5/Relu#encoder/means/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€

s
$encoder/means/BiasAdd/ReadVariableOpReadVariableOpencoder/means/bias*
dtype0*
_output_shapes
:

О
encoder/means/BiasAddBiasAddencoder/means/MatMul$encoder/means/BiasAdd/ReadVariableOp*'
_output_shapes
:€€€€€€€€€
*
T0
≥
7encoder/log_var/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@encoder/log_var/kernel*
valueB"   
   *
dtype0*
_output_shapes
:
•
5encoder/log_var/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *)
_class
loc:@encoder/log_var/kernel*
valueB
 *Ў Њ
•
5encoder/log_var/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@encoder/log_var/kernel*
valueB
 *Ў >*
dtype0*
_output_shapes
: 
л
?encoder/log_var/kernel/Initializer/random_uniform/RandomUniformRandomUniform7encoder/log_var/kernel/Initializer/random_uniform/shape*
_output_shapes
:	А
*
T0*)
_class
loc:@encoder/log_var/kernel*
dtype0
ц
5encoder/log_var/kernel/Initializer/random_uniform/subSub5encoder/log_var/kernel/Initializer/random_uniform/max5encoder/log_var/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@encoder/log_var/kernel*
_output_shapes
: 
Й
5encoder/log_var/kernel/Initializer/random_uniform/mulMul?encoder/log_var/kernel/Initializer/random_uniform/RandomUniform5encoder/log_var/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@encoder/log_var/kernel*
_output_shapes
:	А

ы
1encoder/log_var/kernel/Initializer/random_uniformAdd5encoder/log_var/kernel/Initializer/random_uniform/mul5encoder/log_var/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@encoder/log_var/kernel*
_output_shapes
:	А

і
encoder/log_var/kernelVarHandleOp*)
_class
loc:@encoder/log_var/kernel*
dtype0*
_output_shapes
: *
shape:	А
*'
shared_nameencoder/log_var/kernel
}
7encoder/log_var/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/log_var/kernel*
_output_shapes
: 
і
encoder/log_var/kernel/AssignAssignVariableOpencoder/log_var/kernel1encoder/log_var/kernel/Initializer/random_uniform*)
_class
loc:@encoder/log_var/kernel*
dtype0
≠
*encoder/log_var/kernel/Read/ReadVariableOpReadVariableOpencoder/log_var/kernel*)
_class
loc:@encoder/log_var/kernel*
dtype0*
_output_shapes
:	А

Ь
&encoder/log_var/bias/Initializer/zerosConst*'
_class
loc:@encoder/log_var/bias*
valueB
*    *
dtype0*
_output_shapes
:

©
encoder/log_var/biasVarHandleOp*'
_class
loc:@encoder/log_var/bias*
dtype0*
_output_shapes
: *
shape:
*%
shared_nameencoder/log_var/bias
y
5encoder/log_var/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/log_var/bias*
_output_shapes
: 
£
encoder/log_var/bias/AssignAssignVariableOpencoder/log_var/bias&encoder/log_var/bias/Initializer/zeros*
dtype0*'
_class
loc:@encoder/log_var/bias
Ґ
(encoder/log_var/bias/Read/ReadVariableOpReadVariableOpencoder/log_var/bias*'
_class
loc:@encoder/log_var/bias*
dtype0*
_output_shapes
:

}
%encoder/log_var/MatMul/ReadVariableOpReadVariableOpencoder/log_var/kernel*
dtype0*
_output_shapes
:	А

К
encoder/log_var/MatMulMatMulencoder/e5/Relu%encoder/log_var/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€

w
&encoder/log_var/BiasAdd/ReadVariableOpReadVariableOpencoder/log_var/bias*
_output_shapes
:
*
dtype0
Ф
encoder/log_var/BiasAddBiasAddencoder/log_var/MatMul&encoder/log_var/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€

N
	truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
h
truedivRealDivencoder/log_var/BiasAdd	truediv/y*
T0*'
_output_shapes
:€€€€€€€€€

E
ExpExptruediv*
T0*'
_output_shapes
:€€€€€€€€€

J
ShapeShapeencoder/means/BiasAdd*
T0*
_output_shapes
:
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
А
"random_normal/RandomStandardNormalRandomStandardNormalShape*
dtype0*'
_output_shapes
:€€€€€€€€€
*
T0
Д
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*'
_output_shapes
:€€€€€€€€€
*
T0
m
random_normalAddrandom_normal/mulrandom_normal/mean*'
_output_shapes
:€€€€€€€€€
*
T0
P
mulMulExprandom_normal*
T0*'
_output_shapes
:€€€€€€€€€

l
sampled_latent_variableAddencoder/means/BiasAddmul*'
_output_shapes
:€€€€€€€€€
*
T0
ѓ
5decoder/dense/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@decoder/dense/kernel*
valueB"
      *
dtype0*
_output_shapes
:
°
3decoder/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *'
_class
loc:@decoder/dense/kernel*
valueB
 *Ў Њ
°
3decoder/dense/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@decoder/dense/kernel*
valueB
 *Ў >*
dtype0*
_output_shapes
: 
е
=decoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5decoder/dense/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@decoder/dense/kernel*
dtype0*
_output_shapes
:	
А
о
3decoder/dense/kernel/Initializer/random_uniform/subSub3decoder/dense/kernel/Initializer/random_uniform/max3decoder/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@decoder/dense/kernel*
_output_shapes
: 
Б
3decoder/dense/kernel/Initializer/random_uniform/mulMul=decoder/dense/kernel/Initializer/random_uniform/RandomUniform3decoder/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@decoder/dense/kernel*
_output_shapes
:	
А
у
/decoder/dense/kernel/Initializer/random_uniformAdd3decoder/dense/kernel/Initializer/random_uniform/mul3decoder/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	
А*
T0*'
_class
loc:@decoder/dense/kernel
Ѓ
decoder/dense/kernelVarHandleOp*'
_class
loc:@decoder/dense/kernel*
dtype0*
_output_shapes
: *
shape:	
А*%
shared_namedecoder/dense/kernel
y
5decoder/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/dense/kernel*
_output_shapes
: 
ђ
decoder/dense/kernel/AssignAssignVariableOpdecoder/dense/kernel/decoder/dense/kernel/Initializer/random_uniform*
dtype0*'
_class
loc:@decoder/dense/kernel
І
(decoder/dense/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense/kernel*'
_class
loc:@decoder/dense/kernel*
dtype0*
_output_shapes
:	
А
Ъ
$decoder/dense/bias/Initializer/zerosConst*%
_class
loc:@decoder/dense/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
§
decoder/dense/biasVarHandleOp*
shape:А*#
shared_namedecoder/dense/bias*%
_class
loc:@decoder/dense/bias*
dtype0*
_output_shapes
: 
u
3decoder/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/dense/bias*
_output_shapes
: 
Ы
decoder/dense/bias/AssignAssignVariableOpdecoder/dense/bias$decoder/dense/bias/Initializer/zeros*%
_class
loc:@decoder/dense/bias*
dtype0
Э
&decoder/dense/bias/Read/ReadVariableOpReadVariableOpdecoder/dense/bias*%
_class
loc:@decoder/dense/bias*
dtype0*
_output_shapes	
:А
y
#decoder/dense/MatMul/ReadVariableOpReadVariableOpdecoder/dense/kernel*
dtype0*
_output_shapes
:	
А
П
decoder/dense/MatMulMatMulsampled_latent_variable#decoder/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
t
$decoder/dense/BiasAdd/ReadVariableOpReadVariableOpdecoder/dense/bias*
dtype0*
_output_shapes	
:А
П
decoder/dense/BiasAddBiasAdddecoder/dense/MatMul$decoder/dense/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
d
decoder/dense/ReluReludecoder/dense/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
≥
7decoder/dense_1/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@decoder/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
•
5decoder/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *)
_class
loc:@decoder/dense_1/kernel*
valueB
 *М7Мљ*
dtype0
•
5decoder/dense_1/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@decoder/dense_1/kernel*
valueB
 *М7М=*
dtype0*
_output_shapes
: 
м
?decoder/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7decoder/dense_1/kernel/Initializer/random_uniform/shape*)
_class
loc:@decoder/dense_1/kernel*
dtype0* 
_output_shapes
:
АА*
T0
ц
5decoder/dense_1/kernel/Initializer/random_uniform/subSub5decoder/dense_1/kernel/Initializer/random_uniform/max5decoder/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@decoder/dense_1/kernel*
_output_shapes
: 
К
5decoder/dense_1/kernel/Initializer/random_uniform/mulMul?decoder/dense_1/kernel/Initializer/random_uniform/RandomUniform5decoder/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@decoder/dense_1/kernel* 
_output_shapes
:
АА
ь
1decoder/dense_1/kernel/Initializer/random_uniformAdd5decoder/dense_1/kernel/Initializer/random_uniform/mul5decoder/dense_1/kernel/Initializer/random_uniform/min*)
_class
loc:@decoder/dense_1/kernel* 
_output_shapes
:
АА*
T0
µ
decoder/dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
АА*'
shared_namedecoder/dense_1/kernel*)
_class
loc:@decoder/dense_1/kernel
}
7decoder/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/dense_1/kernel*
_output_shapes
: 
і
decoder/dense_1/kernel/AssignAssignVariableOpdecoder/dense_1/kernel1decoder/dense_1/kernel/Initializer/random_uniform*)
_class
loc:@decoder/dense_1/kernel*
dtype0
Ѓ
*decoder/dense_1/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_1/kernel*
dtype0* 
_output_shapes
:
АА*)
_class
loc:@decoder/dense_1/kernel
™
6decoder/dense_1/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*'
_class
loc:@decoder/dense_1/bias*
valueB:А
Ъ
,decoder/dense_1/bias/Initializer/zeros/ConstConst*'
_class
loc:@decoder/dense_1/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
г
&decoder/dense_1/bias/Initializer/zerosFill6decoder/dense_1/bias/Initializer/zeros/shape_as_tensor,decoder/dense_1/bias/Initializer/zeros/Const*
T0*'
_class
loc:@decoder/dense_1/bias*
_output_shapes	
:А
™
decoder/dense_1/biasVarHandleOp*
_output_shapes
: *
shape:А*%
shared_namedecoder/dense_1/bias*'
_class
loc:@decoder/dense_1/bias*
dtype0
y
5decoder/dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/dense_1/bias*
_output_shapes
: 
£
decoder/dense_1/bias/AssignAssignVariableOpdecoder/dense_1/bias&decoder/dense_1/bias/Initializer/zeros*'
_class
loc:@decoder/dense_1/bias*
dtype0
£
(decoder/dense_1/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_1/bias*'
_class
loc:@decoder/dense_1/bias*
dtype0*
_output_shapes	
:А
~
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOpdecoder/dense_1/kernel*
dtype0* 
_output_shapes
:
АА
О
decoder/dense_1/MatMulMatMuldecoder/dense/Relu%decoder/dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
x
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/dense_1/bias*
dtype0*
_output_shapes	
:А
Х
decoder/dense_1/BiasAddBiasAdddecoder/dense_1/MatMul&decoder/dense_1/BiasAdd/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
T0
h
decoder/dense_1/ReluReludecoder/dense_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
n
decoder/Reshape/shapeConst*%
valueB"€€€€      @   *
dtype0*
_output_shapes
:
Б
decoder/ReshapeReshapedecoder/dense_1/Reludecoder/Reshape/shape*
T0*/
_output_shapes
:€€€€€€€€€@
Ќ
@decoder/conv2d_transpose/kernel/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
Ј
>decoder/conv2d_transpose/kernel/Initializer/random_uniform/minConst*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
valueB
 *„≥]љ*
dtype0*
_output_shapes
: 
Ј
>decoder/conv2d_transpose/kernel/Initializer/random_uniform/maxConst*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
valueB
 *„≥]=*
dtype0*
_output_shapes
: 
Н
Hdecoder/conv2d_transpose/kernel/Initializer/random_uniform/RandomUniformRandomUniform@decoder/conv2d_transpose/kernel/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0*&
_output_shapes
:@@
Ъ
>decoder/conv2d_transpose/kernel/Initializer/random_uniform/subSub>decoder/conv2d_transpose/kernel/Initializer/random_uniform/max>decoder/conv2d_transpose/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
_output_shapes
: 
і
>decoder/conv2d_transpose/kernel/Initializer/random_uniform/mulMulHdecoder/conv2d_transpose/kernel/Initializer/random_uniform/RandomUniform>decoder/conv2d_transpose/kernel/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*&
_output_shapes
:@@
¶
:decoder/conv2d_transpose/kernel/Initializer/random_uniformAdd>decoder/conv2d_transpose/kernel/Initializer/random_uniform/mul>decoder/conv2d_transpose/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*&
_output_shapes
:@@
÷
decoder/conv2d_transpose/kernelVarHandleOp*
shape:@@*0
shared_name!decoder/conv2d_transpose/kernel*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0*
_output_shapes
: 
П
@decoder/conv2d_transpose/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose/kernel*
_output_shapes
: 
Ў
&decoder/conv2d_transpose/kernel/AssignAssignVariableOpdecoder/conv2d_transpose/kernel:decoder/conv2d_transpose/kernel/Initializer/random_uniform*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0
ѕ
3decoder/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/kernel*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0*&
_output_shapes
:@@
Ѓ
/decoder/conv2d_transpose/bias/Initializer/zerosConst*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
valueB@*    *
dtype0*
_output_shapes
:@
ƒ
decoder/conv2d_transpose/biasVarHandleOp*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0*
_output_shapes
: *
shape:@*.
shared_namedecoder/conv2d_transpose/bias
Л
>decoder/conv2d_transpose/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose/bias*
_output_shapes
: 
«
$decoder/conv2d_transpose/bias/AssignAssignVariableOpdecoder/conv2d_transpose/bias/decoder/conv2d_transpose/bias/Initializer/zeros*
dtype0*0
_class&
$"loc:@decoder/conv2d_transpose/bias
љ
1decoder/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/bias*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0*
_output_shapes
:@
]
decoder/conv2d_transpose/ShapeShapedecoder/Reshape*
T0*
_output_shapes
:
v
,decoder/conv2d_transpose/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.decoder/conv2d_transpose/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.decoder/conv2d_transpose/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
™
&decoder/conv2d_transpose/strided_sliceStridedSlicedecoder/conv2d_transpose/Shape,decoder/conv2d_transpose/strided_slice/stack.decoder/conv2d_transpose/strided_slice/stack_1.decoder/conv2d_transpose/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
x
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0
z
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
≤
(decoder/conv2d_transpose/strided_slice_1StridedSlicedecoder/conv2d_transpose/Shape.decoder/conv2d_transpose/strided_slice_1/stack0decoder/conv2d_transpose/strided_slice_1/stack_10decoder/conv2d_transpose/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
x
.decoder/conv2d_transpose/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
≤
(decoder/conv2d_transpose/strided_slice_2StridedSlicedecoder/conv2d_transpose/Shape.decoder/conv2d_transpose/strided_slice_2/stack0decoder/conv2d_transpose/strided_slice_2/stack_10decoder/conv2d_transpose/strided_slice_2/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
`
decoder/conv2d_transpose/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
О
decoder/conv2d_transpose/mulMul(decoder/conv2d_transpose/strided_slice_1decoder/conv2d_transpose/mul/y*
_output_shapes
: *
T0
b
 decoder/conv2d_transpose/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Т
decoder/conv2d_transpose/mul_1Mul(decoder/conv2d_transpose/strided_slice_2 decoder/conv2d_transpose/mul_1/y*
_output_shapes
: *
T0
b
 decoder/conv2d_transpose/stack/3Const*
value	B :@*
dtype0*
_output_shapes
: 
№
decoder/conv2d_transpose/stackPack&decoder/conv2d_transpose/strided_slicedecoder/conv2d_transpose/muldecoder/conv2d_transpose/mul_1 decoder/conv2d_transpose/stack/3*
T0*
N*
_output_shapes
:
x
.decoder/conv2d_transpose/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
≤
(decoder/conv2d_transpose/strided_slice_3StridedSlicedecoder/conv2d_transpose/stack.decoder/conv2d_transpose/strided_slice_3/stack0decoder/conv2d_transpose/strided_slice_3/stack_10decoder/conv2d_transpose/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
†
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/kernel*
dtype0*&
_output_shapes
:@@
Н
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInputdecoder/conv2d_transpose/stack8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpdecoder/Reshape*
T0*
strides
*/
_output_shapes
:€€€€€€€€€@*
paddingSAME
Й
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/bias*
dtype0*
_output_shapes
:@
Ѕ
 decoder/conv2d_transpose/BiasAddBiasAdd)decoder/conv2d_transpose/conv2d_transpose/decoder/conv2d_transpose/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@
Б
decoder/conv2d_transpose/ReluRelu decoder/conv2d_transpose/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
—
Bdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/shapeConst*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*%
valueB"          @   *
dtype0*
_output_shapes
:
ї
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/minConst*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
valueB
 *  Аљ*
dtype0*
_output_shapes
: 
ї
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/maxConst*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
valueB
 *  А=*
dtype0*
_output_shapes
: 
У
Jdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformBdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/shape*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*&
_output_shapes
: @
Ґ
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/subSub@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/max@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel
Љ
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/mulMulJdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniform@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/sub*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*&
_output_shapes
: @*
T0
Ѓ
<decoder/conv2d_transpose_1/kernel/Initializer/random_uniformAdd@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/mul@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/min*&
_output_shapes
: @*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel
№
!decoder/conv2d_transpose_1/kernelVarHandleOp*
shape: @*2
shared_name#!decoder/conv2d_transpose_1/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*
_output_shapes
: 
У
Bdecoder/conv2d_transpose_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!decoder/conv2d_transpose_1/kernel*
_output_shapes
: 
а
(decoder/conv2d_transpose_1/kernel/AssignAssignVariableOp!decoder/conv2d_transpose_1/kernel<decoder/conv2d_transpose_1/kernel/Initializer/random_uniform*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0
’
5decoder/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_1/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*&
_output_shapes
: @
≤
1decoder/conv2d_transpose_1/bias/Initializer/zerosConst*
_output_shapes
: *2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
valueB *    *
dtype0
 
decoder/conv2d_transpose_1/biasVarHandleOp*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
dtype0*
_output_shapes
: *
shape: *0
shared_name!decoder/conv2d_transpose_1/bias
П
@decoder/conv2d_transpose_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose_1/bias*
_output_shapes
: 
ѕ
&decoder/conv2d_transpose_1/bias/AssignAssignVariableOpdecoder/conv2d_transpose_1/bias1decoder/conv2d_transpose_1/bias/Initializer/zeros*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
dtype0
√
3decoder/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_1/bias*
dtype0*
_output_shapes
: *2
_class(
&$loc:@decoder/conv2d_transpose_1/bias
m
 decoder/conv2d_transpose_1/ShapeShapedecoder/conv2d_transpose/Relu*
T0*
_output_shapes
:
x
.decoder/conv2d_transpose_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
z
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
(decoder/conv2d_transpose_1/strided_sliceStridedSlice decoder/conv2d_transpose_1/Shape.decoder/conv2d_transpose_1/strided_slice/stack0decoder/conv2d_transpose_1/strided_slice/stack_10decoder/conv2d_transpose_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
z
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice decoder/conv2d_transpose_1/Shape0decoder/conv2d_transpose_1/strided_slice_1/stack2decoder/conv2d_transpose_1/strided_slice_1/stack_12decoder/conv2d_transpose_1/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
z
0decoder/conv2d_transpose_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
|
2decoder/conv2d_transpose_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_1/strided_slice_2StridedSlice decoder/conv2d_transpose_1/Shape0decoder/conv2d_transpose_1/strided_slice_2/stack2decoder/conv2d_transpose_1/strided_slice_2/stack_12decoder/conv2d_transpose_1/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
b
 decoder/conv2d_transpose_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ф
decoder/conv2d_transpose_1/mulMul*decoder/conv2d_transpose_1/strided_slice_1 decoder/conv2d_transpose_1/mul/y*
T0*
_output_shapes
: 
d
"decoder/conv2d_transpose_1/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
 decoder/conv2d_transpose_1/mul_1Mul*decoder/conv2d_transpose_1/strided_slice_2"decoder/conv2d_transpose_1/mul_1/y*
T0*
_output_shapes
: 
d
"decoder/conv2d_transpose_1/stack/3Const*
value	B : *
dtype0*
_output_shapes
: 
ж
 decoder/conv2d_transpose_1/stackPack(decoder/conv2d_transpose_1/strided_slicedecoder/conv2d_transpose_1/mul decoder/conv2d_transpose_1/mul_1"decoder/conv2d_transpose_1/stack/3*
N*
_output_shapes
:*
T0
z
0decoder/conv2d_transpose_1/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
|
2decoder/conv2d_transpose_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_1/strided_slice_3StridedSlice decoder/conv2d_transpose_1/stack0decoder/conv2d_transpose_1/strided_slice_3/stack2decoder/conv2d_transpose_1/strided_slice_3/stack_12decoder/conv2d_transpose_1/strided_slice_3/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
§
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_1/kernel*
dtype0*&
_output_shapes
: @
°
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput decoder/conv2d_transpose_1/stack:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpdecoder/conv2d_transpose/Relu*
strides
*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
T0
Н
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_1/bias*
dtype0*
_output_shapes
: 
«
"decoder/conv2d_transpose_1/BiasAddBiasAdd+decoder/conv2d_transpose_1/conv2d_transpose1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 
Е
decoder/conv2d_transpose_1/ReluRelu"decoder/conv2d_transpose_1/BiasAdd*/
_output_shapes
:€€€€€€€€€ *
T0
—
Bdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/shapeConst*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*%
valueB"              *
dtype0*
_output_shapes
:
ї
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/minConst*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
valueB
 *qƒЬљ*
dtype0*
_output_shapes
: 
ї
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
valueB
 *qƒЬ=
У
Jdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformBdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/shape*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0*&
_output_shapes
:  
Ґ
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/subSub@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/max@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
_output_shapes
: 
Љ
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/mulMulJdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniform@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*&
_output_shapes
:  
Ѓ
<decoder/conv2d_transpose_2/kernel/Initializer/random_uniformAdd@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/mul@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*&
_output_shapes
:  
№
!decoder/conv2d_transpose_2/kernelVarHandleOp*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0*
_output_shapes
: *
shape:  *2
shared_name#!decoder/conv2d_transpose_2/kernel
У
Bdecoder/conv2d_transpose_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!decoder/conv2d_transpose_2/kernel*
_output_shapes
: 
а
(decoder/conv2d_transpose_2/kernel/AssignAssignVariableOp!decoder/conv2d_transpose_2/kernel<decoder/conv2d_transpose_2/kernel/Initializer/random_uniform*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0
’
5decoder/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_2/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0*&
_output_shapes
:  
≤
1decoder/conv2d_transpose_2/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
valueB *    *
dtype0*
_output_shapes
: 
 
decoder/conv2d_transpose_2/biasVarHandleOp*0
shared_name!decoder/conv2d_transpose_2/bias*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0*
_output_shapes
: *
shape: 
П
@decoder/conv2d_transpose_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose_2/bias*
_output_shapes
: 
ѕ
&decoder/conv2d_transpose_2/bias/AssignAssignVariableOpdecoder/conv2d_transpose_2/bias1decoder/conv2d_transpose_2/bias/Initializer/zeros*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0
√
3decoder/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_2/bias*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0*
_output_shapes
: 
o
 decoder/conv2d_transpose_2/ShapeShapedecoder/conv2d_transpose_1/Relu*
T0*
_output_shapes
:
x
.decoder/conv2d_transpose_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
і
(decoder/conv2d_transpose_2/strided_sliceStridedSlice decoder/conv2d_transpose_2/Shape.decoder/conv2d_transpose_2/strided_slice/stack0decoder/conv2d_transpose_2/strided_slice/stack_10decoder/conv2d_transpose_2/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
z
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
|
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice decoder/conv2d_transpose_2/Shape0decoder/conv2d_transpose_2/strided_slice_1/stack2decoder/conv2d_transpose_2/strided_slice_1/stack_12decoder/conv2d_transpose_2/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
z
0decoder/conv2d_transpose_2/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_2/strided_slice_2StridedSlice decoder/conv2d_transpose_2/Shape0decoder/conv2d_transpose_2/strided_slice_2/stack2decoder/conv2d_transpose_2/strided_slice_2/stack_12decoder/conv2d_transpose_2/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
b
 decoder/conv2d_transpose_2/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ф
decoder/conv2d_transpose_2/mulMul*decoder/conv2d_transpose_2/strided_slice_1 decoder/conv2d_transpose_2/mul/y*
_output_shapes
: *
T0
d
"decoder/conv2d_transpose_2/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
 decoder/conv2d_transpose_2/mul_1Mul*decoder/conv2d_transpose_2/strided_slice_2"decoder/conv2d_transpose_2/mul_1/y*
_output_shapes
: *
T0
d
"decoder/conv2d_transpose_2/stack/3Const*
dtype0*
_output_shapes
: *
value	B : 
ж
 decoder/conv2d_transpose_2/stackPack(decoder/conv2d_transpose_2/strided_slicedecoder/conv2d_transpose_2/mul decoder/conv2d_transpose_2/mul_1"decoder/conv2d_transpose_2/stack/3*
T0*
N*
_output_shapes
:
z
0decoder/conv2d_transpose_2/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
|
2decoder/conv2d_transpose_2/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_2/strided_slice_3StridedSlice decoder/conv2d_transpose_2/stack0decoder/conv2d_transpose_2/strided_slice_3/stack2decoder/conv2d_transpose_2/strided_slice_3/stack_12decoder/conv2d_transpose_2/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
§
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_2/kernel*
dtype0*&
_output_shapes
:  
£
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput decoder/conv2d_transpose_2/stack:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpdecoder/conv2d_transpose_1/Relu*
strides
*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
T0
Н
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_2/bias*
dtype0*
_output_shapes
: 
«
"decoder/conv2d_transpose_2/BiasAddBiasAdd+decoder/conv2d_transpose_2/conv2d_transpose1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€   *
T0
Е
decoder/conv2d_transpose_2/ReluRelu"decoder/conv2d_transpose_2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€   
—
Bdecoder/conv2d_transpose_3/kernel/Initializer/random_uniform/shapeConst*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*%
valueB"             *
dtype0*
_output_shapes
:
ї
@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/minConst*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
valueB
 *чь”љ*
dtype0*
_output_shapes
: 
ї
@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/maxConst*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
valueB
 *чь”=*
dtype0*
_output_shapes
: 
У
Jdecoder/conv2d_transpose_3/kernel/Initializer/random_uniform/RandomUniformRandomUniformBdecoder/conv2d_transpose_3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel
Ґ
@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/subSub@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/max@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
_output_shapes
: 
Љ
@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/mulMulJdecoder/conv2d_transpose_3/kernel/Initializer/random_uniform/RandomUniform@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/sub*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*&
_output_shapes
: *
T0
Ѓ
<decoder/conv2d_transpose_3/kernel/Initializer/random_uniformAdd@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/mul@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/min*&
_output_shapes
: *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel
№
!decoder/conv2d_transpose_3/kernelVarHandleOp*
shape: *2
shared_name#!decoder/conv2d_transpose_3/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
dtype0*
_output_shapes
: 
У
Bdecoder/conv2d_transpose_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!decoder/conv2d_transpose_3/kernel*
_output_shapes
: 
а
(decoder/conv2d_transpose_3/kernel/AssignAssignVariableOp!decoder/conv2d_transpose_3/kernel<decoder/conv2d_transpose_3/kernel/Initializer/random_uniform*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
dtype0
’
5decoder/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_3/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
dtype0*&
_output_shapes
: 
≤
1decoder/conv2d_transpose_3/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/conv2d_transpose_3/bias*
valueB*    *
dtype0*
_output_shapes
:
 
decoder/conv2d_transpose_3/biasVarHandleOp*
shape:*0
shared_name!decoder/conv2d_transpose_3/bias*2
_class(
&$loc:@decoder/conv2d_transpose_3/bias*
dtype0*
_output_shapes
: 
П
@decoder/conv2d_transpose_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose_3/bias*
_output_shapes
: 
ѕ
&decoder/conv2d_transpose_3/bias/AssignAssignVariableOpdecoder/conv2d_transpose_3/bias1decoder/conv2d_transpose_3/bias/Initializer/zeros*2
_class(
&$loc:@decoder/conv2d_transpose_3/bias*
dtype0
√
3decoder/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_3/bias*2
_class(
&$loc:@decoder/conv2d_transpose_3/bias*
dtype0*
_output_shapes
:
o
 decoder/conv2d_transpose_3/ShapeShapedecoder/conv2d_transpose_2/Relu*
_output_shapes
:*
T0
x
.decoder/conv2d_transpose_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
z
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
(decoder/conv2d_transpose_3/strided_sliceStridedSlice decoder/conv2d_transpose_3/Shape.decoder/conv2d_transpose_3/strided_slice/stack0decoder/conv2d_transpose_3/strided_slice/stack_10decoder/conv2d_transpose_3/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
z
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0
|
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice decoder/conv2d_transpose_3/Shape0decoder/conv2d_transpose_3/strided_slice_1/stack2decoder/conv2d_transpose_3/strided_slice_1/stack_12decoder/conv2d_transpose_3/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
z
0decoder/conv2d_transpose_3/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
|
2decoder/conv2d_transpose_3/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_3/strided_slice_2StridedSlice decoder/conv2d_transpose_3/Shape0decoder/conv2d_transpose_3/strided_slice_2/stack2decoder/conv2d_transpose_3/strided_slice_2/stack_12decoder/conv2d_transpose_3/strided_slice_2/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
b
 decoder/conv2d_transpose_3/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ф
decoder/conv2d_transpose_3/mulMul*decoder/conv2d_transpose_3/strided_slice_1 decoder/conv2d_transpose_3/mul/y*
T0*
_output_shapes
: 
d
"decoder/conv2d_transpose_3/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
 decoder/conv2d_transpose_3/mul_1Mul*decoder/conv2d_transpose_3/strided_slice_2"decoder/conv2d_transpose_3/mul_1/y*
T0*
_output_shapes
: 
d
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
value	B :*
dtype0
ж
 decoder/conv2d_transpose_3/stackPack(decoder/conv2d_transpose_3/strided_slicedecoder/conv2d_transpose_3/mul decoder/conv2d_transpose_3/mul_1"decoder/conv2d_transpose_3/stack/3*
T0*
N*
_output_shapes
:
z
0decoder/conv2d_transpose_3/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_3/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
|
2decoder/conv2d_transpose_3/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_3/strided_slice_3StridedSlice decoder/conv2d_transpose_3/stack0decoder/conv2d_transpose_3/strided_slice_3/stack2decoder/conv2d_transpose_3/strided_slice_3/stack_12decoder/conv2d_transpose_3/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
§
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_3/kernel*
dtype0*&
_output_shapes
: 
£
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput decoder/conv2d_transpose_3/stack:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpdecoder/conv2d_transpose_2/Relu*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€@@
Н
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_3/bias*
dtype0*
_output_shapes
:
«
"decoder/conv2d_transpose_3/BiasAddBiasAdd+decoder/conv2d_transpose_3/conv2d_transpose1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@
p
decoder/Reshape_1/shapeConst*
_output_shapes
:*%
valueB"€€€€@   @      *
dtype0
У
decoder/Reshape_1Reshape"decoder/conv2d_transpose_3/BiasAdddecoder/Reshape_1/shape*
T0*/
_output_shapes
:€€€€€€€€€@@
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€
*
shape:€€€€€€€€€

{
%decoder_1/dense/MatMul/ReadVariableOpReadVariableOpdecoder/dense/kernel*
dtype0*
_output_shapes
:	
А
Й
decoder_1/dense/MatMulMatMulPlaceholder_1%decoder_1/dense/MatMul/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
T0
v
&decoder_1/dense/BiasAdd/ReadVariableOpReadVariableOpdecoder/dense/bias*
dtype0*
_output_shapes	
:А
Х
decoder_1/dense/BiasAddBiasAdddecoder_1/dense/MatMul&decoder_1/dense/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
h
decoder_1/dense/ReluReludecoder_1/dense/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
А
'decoder_1/dense_1/MatMul/ReadVariableOpReadVariableOpdecoder/dense_1/kernel*
dtype0* 
_output_shapes
:
АА
Ф
decoder_1/dense_1/MatMulMatMuldecoder_1/dense/Relu'decoder_1/dense_1/MatMul/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
T0
z
(decoder_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/dense_1/bias*
dtype0*
_output_shapes	
:А
Ы
decoder_1/dense_1/BiasAddBiasAdddecoder_1/dense_1/MatMul(decoder_1/dense_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
l
decoder_1/dense_1/ReluReludecoder_1/dense_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
p
decoder_1/Reshape/shapeConst*
_output_shapes
:*%
valueB"€€€€      @   *
dtype0
З
decoder_1/ReshapeReshapedecoder_1/dense_1/Reludecoder_1/Reshape/shape*/
_output_shapes
:€€€€€€€€€@*
T0
a
 decoder_1/conv2d_transpose/ShapeShapedecoder_1/Reshape*
T0*
_output_shapes
:
x
.decoder_1/conv2d_transpose/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder_1/conv2d_transpose/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder_1/conv2d_transpose/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
і
(decoder_1/conv2d_transpose/strided_sliceStridedSlice decoder_1/conv2d_transpose/Shape.decoder_1/conv2d_transpose/strided_slice/stack0decoder_1/conv2d_transpose/strided_slice/stack_10decoder_1/conv2d_transpose/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
z
0decoder_1/conv2d_transpose/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder_1/conv2d_transpose/strided_slice_1StridedSlice decoder_1/conv2d_transpose/Shape0decoder_1/conv2d_transpose/strided_slice_1/stack2decoder_1/conv2d_transpose/strided_slice_1/stack_12decoder_1/conv2d_transpose/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
z
0decoder_1/conv2d_transpose/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Љ
*decoder_1/conv2d_transpose/strided_slice_2StridedSlice decoder_1/conv2d_transpose/Shape0decoder_1/conv2d_transpose/strided_slice_2/stack2decoder_1/conv2d_transpose/strided_slice_2/stack_12decoder_1/conv2d_transpose/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
b
 decoder_1/conv2d_transpose/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ф
decoder_1/conv2d_transpose/mulMul*decoder_1/conv2d_transpose/strided_slice_1 decoder_1/conv2d_transpose/mul/y*
_output_shapes
: *
T0
d
"decoder_1/conv2d_transpose/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
 decoder_1/conv2d_transpose/mul_1Mul*decoder_1/conv2d_transpose/strided_slice_2"decoder_1/conv2d_transpose/mul_1/y*
T0*
_output_shapes
: 
d
"decoder_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
value	B :@*
dtype0
ж
 decoder_1/conv2d_transpose/stackPack(decoder_1/conv2d_transpose/strided_slicedecoder_1/conv2d_transpose/mul decoder_1/conv2d_transpose/mul_1"decoder_1/conv2d_transpose/stack/3*
T0*
N*
_output_shapes
:
z
0decoder_1/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
valueB: *
dtype0
|
2decoder_1/conv2d_transpose/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder_1/conv2d_transpose/strided_slice_3StridedSlice decoder_1/conv2d_transpose/stack0decoder_1/conv2d_transpose/strided_slice_3/stack2decoder_1/conv2d_transpose/strided_slice_3/stack_12decoder_1/conv2d_transpose/strided_slice_3/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
Ґ
:decoder_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/kernel*
dtype0*&
_output_shapes
:@@
Х
+decoder_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput decoder_1/conv2d_transpose/stack:decoder_1/conv2d_transpose/conv2d_transpose/ReadVariableOpdecoder_1/Reshape*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€@
Л
1decoder_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/bias*
dtype0*
_output_shapes
:@
«
"decoder_1/conv2d_transpose/BiasAddBiasAdd+decoder_1/conv2d_transpose/conv2d_transpose1decoder_1/conv2d_transpose/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@
Е
decoder_1/conv2d_transpose/ReluRelu"decoder_1/conv2d_transpose/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
q
"decoder_1/conv2d_transpose_1/ShapeShapedecoder_1/conv2d_transpose/Relu*
_output_shapes
:*
T0
z
0decoder_1/conv2d_transpose_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
|
2decoder_1/conv2d_transpose_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Њ
*decoder_1/conv2d_transpose_1/strided_sliceStridedSlice"decoder_1/conv2d_transpose_1/Shape0decoder_1/conv2d_transpose_1/strided_slice/stack2decoder_1/conv2d_transpose_1/strided_slice/stack_12decoder_1/conv2d_transpose_1/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
|
2decoder_1/conv2d_transpose_1/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
~
4decoder_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_1/strided_slice_1StridedSlice"decoder_1/conv2d_transpose_1/Shape2decoder_1/conv2d_transpose_1/strided_slice_1/stack4decoder_1/conv2d_transpose_1/strided_slice_1/stack_14decoder_1/conv2d_transpose_1/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
|
2decoder_1/conv2d_transpose_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
~
4decoder_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_1/strided_slice_2StridedSlice"decoder_1/conv2d_transpose_1/Shape2decoder_1/conv2d_transpose_1/strided_slice_2/stack4decoder_1/conv2d_transpose_1/strided_slice_2/stack_14decoder_1/conv2d_transpose_1/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
d
"decoder_1/conv2d_transpose_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ъ
 decoder_1/conv2d_transpose_1/mulMul,decoder_1/conv2d_transpose_1/strided_slice_1"decoder_1/conv2d_transpose_1/mul/y*
_output_shapes
: *
T0
f
$decoder_1/conv2d_transpose_1/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ю
"decoder_1/conv2d_transpose_1/mul_1Mul,decoder_1/conv2d_transpose_1/strided_slice_2$decoder_1/conv2d_transpose_1/mul_1/y*
_output_shapes
: *
T0
f
$decoder_1/conv2d_transpose_1/stack/3Const*
value	B : *
dtype0*
_output_shapes
: 
р
"decoder_1/conv2d_transpose_1/stackPack*decoder_1/conv2d_transpose_1/strided_slice decoder_1/conv2d_transpose_1/mul"decoder_1/conv2d_transpose_1/mul_1$decoder_1/conv2d_transpose_1/stack/3*
_output_shapes
:*
T0*
N
|
2decoder_1/conv2d_transpose_1/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
~
4decoder_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_1/strided_slice_3StridedSlice"decoder_1/conv2d_transpose_1/stack2decoder_1/conv2d_transpose_1/strided_slice_3/stack4decoder_1/conv2d_transpose_1/strided_slice_3/stack_14decoder_1/conv2d_transpose_1/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
¶
<decoder_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_1/kernel*&
_output_shapes
: @*
dtype0
©
-decoder_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput"decoder_1/conv2d_transpose_1/stack<decoder_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpdecoder_1/conv2d_transpose/Relu*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€ 
П
3decoder_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_1/bias*
_output_shapes
: *
dtype0
Ќ
$decoder_1/conv2d_transpose_1/BiasAddBiasAdd-decoder_1/conv2d_transpose_1/conv2d_transpose3decoder_1/conv2d_transpose_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 
Й
!decoder_1/conv2d_transpose_1/ReluRelu$decoder_1/conv2d_transpose_1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€ 
s
"decoder_1/conv2d_transpose_2/ShapeShape!decoder_1/conv2d_transpose_1/Relu*
T0*
_output_shapes
:
z
0decoder_1/conv2d_transpose_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
|
2decoder_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Њ
*decoder_1/conv2d_transpose_2/strided_sliceStridedSlice"decoder_1/conv2d_transpose_2/Shape0decoder_1/conv2d_transpose_2/strided_slice/stack2decoder_1/conv2d_transpose_2/strided_slice/stack_12decoder_1/conv2d_transpose_2/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
|
2decoder_1/conv2d_transpose_2/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
~
4decoder_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
∆
,decoder_1/conv2d_transpose_2/strided_slice_1StridedSlice"decoder_1/conv2d_transpose_2/Shape2decoder_1/conv2d_transpose_2/strided_slice_1/stack4decoder_1/conv2d_transpose_2/strided_slice_1/stack_14decoder_1/conv2d_transpose_2/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
|
2decoder_1/conv2d_transpose_2/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
∆
,decoder_1/conv2d_transpose_2/strided_slice_2StridedSlice"decoder_1/conv2d_transpose_2/Shape2decoder_1/conv2d_transpose_2/strided_slice_2/stack4decoder_1/conv2d_transpose_2/strided_slice_2/stack_14decoder_1/conv2d_transpose_2/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
d
"decoder_1/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
value	B :*
dtype0
Ъ
 decoder_1/conv2d_transpose_2/mulMul,decoder_1/conv2d_transpose_2/strided_slice_1"decoder_1/conv2d_transpose_2/mul/y*
_output_shapes
: *
T0
f
$decoder_1/conv2d_transpose_2/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :
Ю
"decoder_1/conv2d_transpose_2/mul_1Mul,decoder_1/conv2d_transpose_2/strided_slice_2$decoder_1/conv2d_transpose_2/mul_1/y*
T0*
_output_shapes
: 
f
$decoder_1/conv2d_transpose_2/stack/3Const*
dtype0*
_output_shapes
: *
value	B : 
р
"decoder_1/conv2d_transpose_2/stackPack*decoder_1/conv2d_transpose_2/strided_slice decoder_1/conv2d_transpose_2/mul"decoder_1/conv2d_transpose_2/mul_1$decoder_1/conv2d_transpose_2/stack/3*
T0*
N*
_output_shapes
:
|
2decoder_1/conv2d_transpose_2/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB: 
~
4decoder_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
~
4decoder_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_2/strided_slice_3StridedSlice"decoder_1/conv2d_transpose_2/stack2decoder_1/conv2d_transpose_2/strided_slice_3/stack4decoder_1/conv2d_transpose_2/strided_slice_3/stack_14decoder_1/conv2d_transpose_2/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
¶
<decoder_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_2/kernel*
dtype0*&
_output_shapes
:  
Ђ
-decoder_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput"decoder_1/conv2d_transpose_2/stack<decoder_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp!decoder_1/conv2d_transpose_1/Relu*
strides
*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
T0
П
3decoder_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_2/bias*
dtype0*
_output_shapes
: 
Ќ
$decoder_1/conv2d_transpose_2/BiasAddBiasAdd-decoder_1/conv2d_transpose_2/conv2d_transpose3decoder_1/conv2d_transpose_2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   
Й
!decoder_1/conv2d_transpose_2/ReluRelu$decoder_1/conv2d_transpose_2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€   
s
"decoder_1/conv2d_transpose_3/ShapeShape!decoder_1/conv2d_transpose_2/Relu*
T0*
_output_shapes
:
z
0decoder_1/conv2d_transpose_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Њ
*decoder_1/conv2d_transpose_3/strided_sliceStridedSlice"decoder_1/conv2d_transpose_3/Shape0decoder_1/conv2d_transpose_3/strided_slice/stack2decoder_1/conv2d_transpose_3/strided_slice/stack_12decoder_1/conv2d_transpose_3/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
|
2decoder_1/conv2d_transpose_3/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
~
4decoder_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_3/strided_slice_1StridedSlice"decoder_1/conv2d_transpose_3/Shape2decoder_1/conv2d_transpose_3/strided_slice_1/stack4decoder_1/conv2d_transpose_3/strided_slice_1/stack_14decoder_1/conv2d_transpose_3/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
|
2decoder_1/conv2d_transpose_3/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_3/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_3/strided_slice_2StridedSlice"decoder_1/conv2d_transpose_3/Shape2decoder_1/conv2d_transpose_3/strided_slice_2/stack4decoder_1/conv2d_transpose_3/strided_slice_2/stack_14decoder_1/conv2d_transpose_3/strided_slice_2/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
d
"decoder_1/conv2d_transpose_3/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ъ
 decoder_1/conv2d_transpose_3/mulMul,decoder_1/conv2d_transpose_3/strided_slice_1"decoder_1/conv2d_transpose_3/mul/y*
T0*
_output_shapes
: 
f
$decoder_1/conv2d_transpose_3/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ю
"decoder_1/conv2d_transpose_3/mul_1Mul,decoder_1/conv2d_transpose_3/strided_slice_2$decoder_1/conv2d_transpose_3/mul_1/y*
T0*
_output_shapes
: 
f
$decoder_1/conv2d_transpose_3/stack/3Const*
value	B :*
dtype0*
_output_shapes
: 
р
"decoder_1/conv2d_transpose_3/stackPack*decoder_1/conv2d_transpose_3/strided_slice decoder_1/conv2d_transpose_3/mul"decoder_1/conv2d_transpose_3/mul_1$decoder_1/conv2d_transpose_3/stack/3*
_output_shapes
:*
T0*
N
|
2decoder_1/conv2d_transpose_3/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_3/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_3/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_3/strided_slice_3StridedSlice"decoder_1/conv2d_transpose_3/stack2decoder_1/conv2d_transpose_3/strided_slice_3/stack4decoder_1/conv2d_transpose_3/strided_slice_3/stack_14decoder_1/conv2d_transpose_3/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
¶
<decoder_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_3/kernel*
dtype0*&
_output_shapes
: 
Ђ
-decoder_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput"decoder_1/conv2d_transpose_3/stack<decoder_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp!decoder_1/conv2d_transpose_2/Relu*/
_output_shapes
:€€€€€€€€€@@*
paddingSAME*
T0*
strides

П
3decoder_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_3/bias*
dtype0*
_output_shapes
:
Ќ
$decoder_1/conv2d_transpose_3/BiasAddBiasAdd-decoder_1/conv2d_transpose_3/conv2d_transpose3decoder_1/conv2d_transpose_3/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@
r
decoder_1/Reshape_1/shapeConst*%
valueB"€€€€@   @      *
dtype0*
_output_shapes
:
Щ
decoder_1/Reshape_1Reshape$decoder_1/conv2d_transpose_3/BiasAdddecoder_1/Reshape_1/shape*
T0*/
_output_shapes
:€€€€€€€€€@@"&"• 
	variablesЧ Ф 
М
encoder/e1/kernel:0encoder/e1/kernel/Assign'encoder/e1/kernel/Read/ReadVariableOp:0(2.encoder/e1/kernel/Initializer/random_uniform:08
{
encoder/e1/bias:0encoder/e1/bias/Assign%encoder/e1/bias/Read/ReadVariableOp:0(2#encoder/e1/bias/Initializer/zeros:08
М
encoder/e2/kernel:0encoder/e2/kernel/Assign'encoder/e2/kernel/Read/ReadVariableOp:0(2.encoder/e2/kernel/Initializer/random_uniform:08
{
encoder/e2/bias:0encoder/e2/bias/Assign%encoder/e2/bias/Read/ReadVariableOp:0(2#encoder/e2/bias/Initializer/zeros:08
М
encoder/e3/kernel:0encoder/e3/kernel/Assign'encoder/e3/kernel/Read/ReadVariableOp:0(2.encoder/e3/kernel/Initializer/random_uniform:08
{
encoder/e3/bias:0encoder/e3/bias/Assign%encoder/e3/bias/Read/ReadVariableOp:0(2#encoder/e3/bias/Initializer/zeros:08
М
encoder/e4/kernel:0encoder/e4/kernel/Assign'encoder/e4/kernel/Read/ReadVariableOp:0(2.encoder/e4/kernel/Initializer/random_uniform:08
{
encoder/e4/bias:0encoder/e4/bias/Assign%encoder/e4/bias/Read/ReadVariableOp:0(2#encoder/e4/bias/Initializer/zeros:08
М
encoder/e5/kernel:0encoder/e5/kernel/Assign'encoder/e5/kernel/Read/ReadVariableOp:0(2.encoder/e5/kernel/Initializer/random_uniform:08
{
encoder/e5/bias:0encoder/e5/bias/Assign%encoder/e5/bias/Read/ReadVariableOp:0(2#encoder/e5/bias/Initializer/zeros:08
Ш
encoder/means/kernel:0encoder/means/kernel/Assign*encoder/means/kernel/Read/ReadVariableOp:0(21encoder/means/kernel/Initializer/random_uniform:08
З
encoder/means/bias:0encoder/means/bias/Assign(encoder/means/bias/Read/ReadVariableOp:0(2&encoder/means/bias/Initializer/zeros:08
†
encoder/log_var/kernel:0encoder/log_var/kernel/Assign,encoder/log_var/kernel/Read/ReadVariableOp:0(23encoder/log_var/kernel/Initializer/random_uniform:08
П
encoder/log_var/bias:0encoder/log_var/bias/Assign*encoder/log_var/bias/Read/ReadVariableOp:0(2(encoder/log_var/bias/Initializer/zeros:08
Ш
decoder/dense/kernel:0decoder/dense/kernel/Assign*decoder/dense/kernel/Read/ReadVariableOp:0(21decoder/dense/kernel/Initializer/random_uniform:08
З
decoder/dense/bias:0decoder/dense/bias/Assign(decoder/dense/bias/Read/ReadVariableOp:0(2&decoder/dense/bias/Initializer/zeros:08
†
decoder/dense_1/kernel:0decoder/dense_1/kernel/Assign,decoder/dense_1/kernel/Read/ReadVariableOp:0(23decoder/dense_1/kernel/Initializer/random_uniform:08
П
decoder/dense_1/bias:0decoder/dense_1/bias/Assign*decoder/dense_1/bias/Read/ReadVariableOp:0(2(decoder/dense_1/bias/Initializer/zeros:08
ƒ
!decoder/conv2d_transpose/kernel:0&decoder/conv2d_transpose/kernel/Assign5decoder/conv2d_transpose/kernel/Read/ReadVariableOp:0(2<decoder/conv2d_transpose/kernel/Initializer/random_uniform:08
≥
decoder/conv2d_transpose/bias:0$decoder/conv2d_transpose/bias/Assign3decoder/conv2d_transpose/bias/Read/ReadVariableOp:0(21decoder/conv2d_transpose/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_1/kernel:0(decoder/conv2d_transpose_1/kernel/Assign7decoder/conv2d_transpose_1/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_1/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_1/bias:0&decoder/conv2d_transpose_1/bias/Assign5decoder/conv2d_transpose_1/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_1/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_2/kernel:0(decoder/conv2d_transpose_2/kernel/Assign7decoder/conv2d_transpose_2/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_2/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_2/bias:0&decoder/conv2d_transpose_2/bias/Assign5decoder/conv2d_transpose_2/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_2/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_3/kernel:0(decoder/conv2d_transpose_3/kernel/Assign7decoder/conv2d_transpose_3/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_3/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_3/bias:0&decoder/conv2d_transpose_3/bias/Assign5decoder/conv2d_transpose_3/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_3/bias/Initializer/zeros:08"ѓ 
trainable_variablesЧ Ф 
М
encoder/e1/kernel:0encoder/e1/kernel/Assign'encoder/e1/kernel/Read/ReadVariableOp:0(2.encoder/e1/kernel/Initializer/random_uniform:08
{
encoder/e1/bias:0encoder/e1/bias/Assign%encoder/e1/bias/Read/ReadVariableOp:0(2#encoder/e1/bias/Initializer/zeros:08
М
encoder/e2/kernel:0encoder/e2/kernel/Assign'encoder/e2/kernel/Read/ReadVariableOp:0(2.encoder/e2/kernel/Initializer/random_uniform:08
{
encoder/e2/bias:0encoder/e2/bias/Assign%encoder/e2/bias/Read/ReadVariableOp:0(2#encoder/e2/bias/Initializer/zeros:08
М
encoder/e3/kernel:0encoder/e3/kernel/Assign'encoder/e3/kernel/Read/ReadVariableOp:0(2.encoder/e3/kernel/Initializer/random_uniform:08
{
encoder/e3/bias:0encoder/e3/bias/Assign%encoder/e3/bias/Read/ReadVariableOp:0(2#encoder/e3/bias/Initializer/zeros:08
М
encoder/e4/kernel:0encoder/e4/kernel/Assign'encoder/e4/kernel/Read/ReadVariableOp:0(2.encoder/e4/kernel/Initializer/random_uniform:08
{
encoder/e4/bias:0encoder/e4/bias/Assign%encoder/e4/bias/Read/ReadVariableOp:0(2#encoder/e4/bias/Initializer/zeros:08
М
encoder/e5/kernel:0encoder/e5/kernel/Assign'encoder/e5/kernel/Read/ReadVariableOp:0(2.encoder/e5/kernel/Initializer/random_uniform:08
{
encoder/e5/bias:0encoder/e5/bias/Assign%encoder/e5/bias/Read/ReadVariableOp:0(2#encoder/e5/bias/Initializer/zeros:08
Ш
encoder/means/kernel:0encoder/means/kernel/Assign*encoder/means/kernel/Read/ReadVariableOp:0(21encoder/means/kernel/Initializer/random_uniform:08
З
encoder/means/bias:0encoder/means/bias/Assign(encoder/means/bias/Read/ReadVariableOp:0(2&encoder/means/bias/Initializer/zeros:08
†
encoder/log_var/kernel:0encoder/log_var/kernel/Assign,encoder/log_var/kernel/Read/ReadVariableOp:0(23encoder/log_var/kernel/Initializer/random_uniform:08
П
encoder/log_var/bias:0encoder/log_var/bias/Assign*encoder/log_var/bias/Read/ReadVariableOp:0(2(encoder/log_var/bias/Initializer/zeros:08
Ш
decoder/dense/kernel:0decoder/dense/kernel/Assign*decoder/dense/kernel/Read/ReadVariableOp:0(21decoder/dense/kernel/Initializer/random_uniform:08
З
decoder/dense/bias:0decoder/dense/bias/Assign(decoder/dense/bias/Read/ReadVariableOp:0(2&decoder/dense/bias/Initializer/zeros:08
†
decoder/dense_1/kernel:0decoder/dense_1/kernel/Assign,decoder/dense_1/kernel/Read/ReadVariableOp:0(23decoder/dense_1/kernel/Initializer/random_uniform:08
П
decoder/dense_1/bias:0decoder/dense_1/bias/Assign*decoder/dense_1/bias/Read/ReadVariableOp:0(2(decoder/dense_1/bias/Initializer/zeros:08
ƒ
!decoder/conv2d_transpose/kernel:0&decoder/conv2d_transpose/kernel/Assign5decoder/conv2d_transpose/kernel/Read/ReadVariableOp:0(2<decoder/conv2d_transpose/kernel/Initializer/random_uniform:08
≥
decoder/conv2d_transpose/bias:0$decoder/conv2d_transpose/bias/Assign3decoder/conv2d_transpose/bias/Read/ReadVariableOp:0(21decoder/conv2d_transpose/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_1/kernel:0(decoder/conv2d_transpose_1/kernel/Assign7decoder/conv2d_transpose_1/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_1/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_1/bias:0&decoder/conv2d_transpose_1/bias/Assign5decoder/conv2d_transpose_1/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_1/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_2/kernel:0(decoder/conv2d_transpose_2/kernel/Assign7decoder/conv2d_transpose_2/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_2/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_2/bias:0&decoder/conv2d_transpose_2/bias/Assign5decoder/conv2d_transpose_2/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_2/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_3/kernel:0(decoder/conv2d_transpose_3/kernel/Assign7decoder/conv2d_transpose_3/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_3/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_3/bias:0&decoder/conv2d_transpose_3/bias/Assign5decoder/conv2d_transpose_3/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_3/bias/Initializer/zeros:08*Ѕ
gaussian_encoderђ
6
images,
Placeholder:0€€€€€€€€€@@:
logvar0
encoder/log_var/BiasAdd:0€€€€€€€€€
6
mean.
encoder/means/BiasAdd:0€€€€€€€€€
*Й
reconstructionsv
6
images,
Placeholder:0€€€€€€€€€@@<
images2
decoder/Reshape_1:0€€€€€€€€€@@*Е
decoderz
8
latent_vectors&
Placeholder_1:0€€€€€€€€€
>
images4
decoder_1/Reshape_1:0€€€€€€€€€@@”И
ж–
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ъ
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

њ
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
ц
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И*1.14.02unknown8і∞
~
PlaceholderPlaceholder*
dtype0*/
_output_shapes
:€€€€€€€€€@@*$
shape:€€€€€€€€€@@
±
2encoder/e1/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@encoder/e1/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Ы
0encoder/e1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *$
_class
loc:@encoder/e1/kernel*
valueB
 *чь”љ
Ы
0encoder/e1/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@encoder/e1/kernel*
valueB
 *чь”=*
dtype0*
_output_shapes
: 
г
:encoder/e1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e1/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@encoder/e1/kernel*
dtype0*&
_output_shapes
: 
в
0encoder/e1/kernel/Initializer/random_uniform/subSub0encoder/e1/kernel/Initializer/random_uniform/max0encoder/e1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e1/kernel*
_output_shapes
: 
ь
0encoder/e1/kernel/Initializer/random_uniform/mulMul:encoder/e1/kernel/Initializer/random_uniform/RandomUniform0encoder/e1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@encoder/e1/kernel*&
_output_shapes
: 
о
,encoder/e1/kernel/Initializer/random_uniformAdd0encoder/e1/kernel/Initializer/random_uniform/mul0encoder/e1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e1/kernel*&
_output_shapes
: 
ђ
encoder/e1/kernelVarHandleOp*
shape: *"
shared_nameencoder/e1/kernel*$
_class
loc:@encoder/e1/kernel*
dtype0*
_output_shapes
: 
s
2encoder/e1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e1/kernel*
_output_shapes
: 
†
encoder/e1/kernel/AssignAssignVariableOpencoder/e1/kernel,encoder/e1/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e1/kernel*
dtype0
•
%encoder/e1/kernel/Read/ReadVariableOpReadVariableOpencoder/e1/kernel*$
_class
loc:@encoder/e1/kernel*
dtype0*&
_output_shapes
: 
Т
!encoder/e1/bias/Initializer/zerosConst*
_output_shapes
: *"
_class
loc:@encoder/e1/bias*
valueB *    *
dtype0
Ъ
encoder/e1/biasVarHandleOp*
_output_shapes
: *
shape: * 
shared_nameencoder/e1/bias*"
_class
loc:@encoder/e1/bias*
dtype0
o
0encoder/e1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e1/bias*
_output_shapes
: 
П
encoder/e1/bias/AssignAssignVariableOpencoder/e1/bias!encoder/e1/bias/Initializer/zeros*"
_class
loc:@encoder/e1/bias*
dtype0
У
#encoder/e1/bias/Read/ReadVariableOpReadVariableOpencoder/e1/bias*"
_class
loc:@encoder/e1/bias*
dtype0*
_output_shapes
: 
i
encoder/e1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
z
 encoder/e1/Conv2D/ReadVariableOpReadVariableOpencoder/e1/kernel*
dtype0*&
_output_shapes
: 
ђ
encoder/e1/Conv2DConv2DPlaceholder encoder/e1/Conv2D/ReadVariableOp*
strides
*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
T0
m
!encoder/e1/BiasAdd/ReadVariableOpReadVariableOpencoder/e1/bias*
dtype0*
_output_shapes
: 
Н
encoder/e1/BiasAddBiasAddencoder/e1/Conv2D!encoder/e1/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€   *
T0
e
encoder/e1/ReluReluencoder/e1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€   
±
2encoder/e2/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@encoder/e2/kernel*%
valueB"              *
dtype0*
_output_shapes
:
Ы
0encoder/e2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e2/kernel*
valueB
 *qƒЬљ*
dtype0*
_output_shapes
: 
Ы
0encoder/e2/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@encoder/e2/kernel*
valueB
 *qƒЬ=*
dtype0*
_output_shapes
: 
г
:encoder/e2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e2/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@encoder/e2/kernel*
dtype0*&
_output_shapes
:  
в
0encoder/e2/kernel/Initializer/random_uniform/subSub0encoder/e2/kernel/Initializer/random_uniform/max0encoder/e2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e2/kernel*
_output_shapes
: 
ь
0encoder/e2/kernel/Initializer/random_uniform/mulMul:encoder/e2/kernel/Initializer/random_uniform/RandomUniform0encoder/e2/kernel/Initializer/random_uniform/sub*$
_class
loc:@encoder/e2/kernel*&
_output_shapes
:  *
T0
о
,encoder/e2/kernel/Initializer/random_uniformAdd0encoder/e2/kernel/Initializer/random_uniform/mul0encoder/e2/kernel/Initializer/random_uniform/min*&
_output_shapes
:  *
T0*$
_class
loc:@encoder/e2/kernel
ђ
encoder/e2/kernelVarHandleOp*
_output_shapes
: *
shape:  *"
shared_nameencoder/e2/kernel*$
_class
loc:@encoder/e2/kernel*
dtype0
s
2encoder/e2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e2/kernel*
_output_shapes
: 
†
encoder/e2/kernel/AssignAssignVariableOpencoder/e2/kernel,encoder/e2/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e2/kernel*
dtype0
•
%encoder/e2/kernel/Read/ReadVariableOpReadVariableOpencoder/e2/kernel*$
_class
loc:@encoder/e2/kernel*
dtype0*&
_output_shapes
:  
Т
!encoder/e2/bias/Initializer/zerosConst*"
_class
loc:@encoder/e2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Ъ
encoder/e2/biasVarHandleOp*"
_class
loc:@encoder/e2/bias*
dtype0*
_output_shapes
: *
shape: * 
shared_nameencoder/e2/bias
o
0encoder/e2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e2/bias*
_output_shapes
: 
П
encoder/e2/bias/AssignAssignVariableOpencoder/e2/bias!encoder/e2/bias/Initializer/zeros*"
_class
loc:@encoder/e2/bias*
dtype0
У
#encoder/e2/bias/Read/ReadVariableOpReadVariableOpencoder/e2/bias*
_output_shapes
: *"
_class
loc:@encoder/e2/bias*
dtype0
i
encoder/e2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
z
 encoder/e2/Conv2D/ReadVariableOpReadVariableOpencoder/e2/kernel*&
_output_shapes
:  *
dtype0
∞
encoder/e2/Conv2DConv2Dencoder/e1/Relu encoder/e2/Conv2D/ReadVariableOp*
strides
*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
T0
m
!encoder/e2/BiasAdd/ReadVariableOpReadVariableOpencoder/e2/bias*
dtype0*
_output_shapes
: 
Н
encoder/e2/BiasAddBiasAddencoder/e2/Conv2D!encoder/e2/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€ *
T0
e
encoder/e2/ReluReluencoder/e2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€ 
±
2encoder/e3/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@encoder/e3/kernel*%
valueB"          @   *
dtype0*
_output_shapes
:
Ы
0encoder/e3/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e3/kernel*
valueB
 *   Њ*
dtype0*
_output_shapes
: 
Ы
0encoder/e3/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@encoder/e3/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
г
:encoder/e3/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: @*
T0*$
_class
loc:@encoder/e3/kernel
в
0encoder/e3/kernel/Initializer/random_uniform/subSub0encoder/e3/kernel/Initializer/random_uniform/max0encoder/e3/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e3/kernel*
_output_shapes
: 
ь
0encoder/e3/kernel/Initializer/random_uniform/mulMul:encoder/e3/kernel/Initializer/random_uniform/RandomUniform0encoder/e3/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@encoder/e3/kernel*&
_output_shapes
: @
о
,encoder/e3/kernel/Initializer/random_uniformAdd0encoder/e3/kernel/Initializer/random_uniform/mul0encoder/e3/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e3/kernel*&
_output_shapes
: @
ђ
encoder/e3/kernelVarHandleOp*$
_class
loc:@encoder/e3/kernel*
dtype0*
_output_shapes
: *
shape: @*"
shared_nameencoder/e3/kernel
s
2encoder/e3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e3/kernel*
_output_shapes
: 
†
encoder/e3/kernel/AssignAssignVariableOpencoder/e3/kernel,encoder/e3/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e3/kernel*
dtype0
•
%encoder/e3/kernel/Read/ReadVariableOpReadVariableOpencoder/e3/kernel*$
_class
loc:@encoder/e3/kernel*
dtype0*&
_output_shapes
: @
Т
!encoder/e3/bias/Initializer/zerosConst*"
_class
loc:@encoder/e3/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ъ
encoder/e3/biasVarHandleOp*"
_class
loc:@encoder/e3/bias*
dtype0*
_output_shapes
: *
shape:@* 
shared_nameencoder/e3/bias
o
0encoder/e3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e3/bias*
_output_shapes
: 
П
encoder/e3/bias/AssignAssignVariableOpencoder/e3/bias!encoder/e3/bias/Initializer/zeros*"
_class
loc:@encoder/e3/bias*
dtype0
У
#encoder/e3/bias/Read/ReadVariableOpReadVariableOpencoder/e3/bias*"
_class
loc:@encoder/e3/bias*
dtype0*
_output_shapes
:@
i
encoder/e3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
z
 encoder/e3/Conv2D/ReadVariableOpReadVariableOpencoder/e3/kernel*
dtype0*&
_output_shapes
: @
∞
encoder/e3/Conv2DConv2Dencoder/e2/Relu encoder/e3/Conv2D/ReadVariableOp*
strides
*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
T0
m
!encoder/e3/BiasAdd/ReadVariableOpReadVariableOpencoder/e3/bias*
dtype0*
_output_shapes
:@
Н
encoder/e3/BiasAddBiasAddencoder/e3/Conv2D!encoder/e3/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€@*
T0
e
encoder/e3/ReluReluencoder/e3/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
±
2encoder/e4/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@encoder/e4/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
Ы
0encoder/e4/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e4/kernel*
valueB
 *„≥Ёљ*
dtype0*
_output_shapes
: 
Ы
0encoder/e4/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@encoder/e4/kernel*
valueB
 *„≥Ё=*
dtype0*
_output_shapes
: 
г
:encoder/e4/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e4/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@encoder/e4/kernel*
dtype0*&
_output_shapes
:@@
в
0encoder/e4/kernel/Initializer/random_uniform/subSub0encoder/e4/kernel/Initializer/random_uniform/max0encoder/e4/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e4/kernel*
_output_shapes
: 
ь
0encoder/e4/kernel/Initializer/random_uniform/mulMul:encoder/e4/kernel/Initializer/random_uniform/RandomUniform0encoder/e4/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@encoder/e4/kernel*&
_output_shapes
:@@
о
,encoder/e4/kernel/Initializer/random_uniformAdd0encoder/e4/kernel/Initializer/random_uniform/mul0encoder/e4/kernel/Initializer/random_uniform/min*&
_output_shapes
:@@*
T0*$
_class
loc:@encoder/e4/kernel
ђ
encoder/e4/kernelVarHandleOp*$
_class
loc:@encoder/e4/kernel*
dtype0*
_output_shapes
: *
shape:@@*"
shared_nameencoder/e4/kernel
s
2encoder/e4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e4/kernel*
_output_shapes
: 
†
encoder/e4/kernel/AssignAssignVariableOpencoder/e4/kernel,encoder/e4/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e4/kernel*
dtype0
•
%encoder/e4/kernel/Read/ReadVariableOpReadVariableOpencoder/e4/kernel*
dtype0*&
_output_shapes
:@@*$
_class
loc:@encoder/e4/kernel
Т
!encoder/e4/bias/Initializer/zerosConst*
_output_shapes
:@*"
_class
loc:@encoder/e4/bias*
valueB@*    *
dtype0
Ъ
encoder/e4/biasVarHandleOp*"
_class
loc:@encoder/e4/bias*
dtype0*
_output_shapes
: *
shape:@* 
shared_nameencoder/e4/bias
o
0encoder/e4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e4/bias*
_output_shapes
: 
П
encoder/e4/bias/AssignAssignVariableOpencoder/e4/bias!encoder/e4/bias/Initializer/zeros*"
_class
loc:@encoder/e4/bias*
dtype0
У
#encoder/e4/bias/Read/ReadVariableOpReadVariableOpencoder/e4/bias*"
_class
loc:@encoder/e4/bias*
dtype0*
_output_shapes
:@
i
encoder/e4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
z
 encoder/e4/Conv2D/ReadVariableOpReadVariableOpencoder/e4/kernel*
dtype0*&
_output_shapes
:@@
∞
encoder/e4/Conv2DConv2Dencoder/e3/Relu encoder/e4/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€@
m
!encoder/e4/BiasAdd/ReadVariableOpReadVariableOpencoder/e4/bias*
dtype0*
_output_shapes
:@
Н
encoder/e4/BiasAddBiasAddencoder/e4/Conv2D!encoder/e4/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@
e
encoder/e4/ReluReluencoder/e4/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
T
encoder/flatten/ShapeShapeencoder/e4/Relu*
_output_shapes
:*
T0
m
#encoder/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%encoder/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%encoder/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
э
encoder/flatten/strided_sliceStridedSliceencoder/flatten/Shape#encoder/flatten/strided_slice/stack%encoder/flatten/strided_slice/stack_1%encoder/flatten/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
j
encoder/flatten/Reshape/shape/1Const*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
У
encoder/flatten/Reshape/shapePackencoder/flatten/strided_sliceencoder/flatten/Reshape/shape/1*
_output_shapes
:*
T0*
N
Е
encoder/flatten/ReshapeReshapeencoder/e4/Reluencoder/flatten/Reshape/shape*(
_output_shapes
:€€€€€€€€€А*
T0
©
2encoder/e5/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@encoder/e5/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ы
0encoder/e5/kernel/Initializer/random_uniform/minConst*$
_class
loc:@encoder/e5/kernel*
valueB
 *М7Мљ*
dtype0*
_output_shapes
: 
Ы
0encoder/e5/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@encoder/e5/kernel*
valueB
 *М7М=*
dtype0*
_output_shapes
: 
Ё
:encoder/e5/kernel/Initializer/random_uniform/RandomUniformRandomUniform2encoder/e5/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@encoder/e5/kernel*
dtype0* 
_output_shapes
:
АА
в
0encoder/e5/kernel/Initializer/random_uniform/subSub0encoder/e5/kernel/Initializer/random_uniform/max0encoder/e5/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e5/kernel*
_output_shapes
: 
ц
0encoder/e5/kernel/Initializer/random_uniform/mulMul:encoder/e5/kernel/Initializer/random_uniform/RandomUniform0encoder/e5/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@encoder/e5/kernel* 
_output_shapes
:
АА
и
,encoder/e5/kernel/Initializer/random_uniformAdd0encoder/e5/kernel/Initializer/random_uniform/mul0encoder/e5/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@encoder/e5/kernel* 
_output_shapes
:
АА
¶
encoder/e5/kernelVarHandleOp*
_output_shapes
: *
shape:
АА*"
shared_nameencoder/e5/kernel*$
_class
loc:@encoder/e5/kernel*
dtype0
s
2encoder/e5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e5/kernel*
_output_shapes
: 
†
encoder/e5/kernel/AssignAssignVariableOpencoder/e5/kernel,encoder/e5/kernel/Initializer/random_uniform*$
_class
loc:@encoder/e5/kernel*
dtype0
Я
%encoder/e5/kernel/Read/ReadVariableOpReadVariableOpencoder/e5/kernel*$
_class
loc:@encoder/e5/kernel*
dtype0* 
_output_shapes
:
АА
Ф
!encoder/e5/bias/Initializer/zerosConst*"
_class
loc:@encoder/e5/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ы
encoder/e5/biasVarHandleOp*
shape:А* 
shared_nameencoder/e5/bias*"
_class
loc:@encoder/e5/bias*
dtype0*
_output_shapes
: 
o
0encoder/e5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/e5/bias*
_output_shapes
: 
П
encoder/e5/bias/AssignAssignVariableOpencoder/e5/bias!encoder/e5/bias/Initializer/zeros*"
_class
loc:@encoder/e5/bias*
dtype0
Ф
#encoder/e5/bias/Read/ReadVariableOpReadVariableOpencoder/e5/bias*"
_class
loc:@encoder/e5/bias*
dtype0*
_output_shapes	
:А
t
 encoder/e5/MatMul/ReadVariableOpReadVariableOpencoder/e5/kernel*
dtype0* 
_output_shapes
:
АА
Й
encoder/e5/MatMulMatMulencoder/flatten/Reshape encoder/e5/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
n
!encoder/e5/BiasAdd/ReadVariableOpReadVariableOpencoder/e5/bias*
dtype0*
_output_shapes	
:А
Ж
encoder/e5/BiasAddBiasAddencoder/e5/MatMul!encoder/e5/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
^
encoder/e5/ReluReluencoder/e5/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
ѓ
5encoder/means/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@encoder/means/kernel*
valueB"   
   *
dtype0*
_output_shapes
:
°
3encoder/means/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *'
_class
loc:@encoder/means/kernel*
valueB
 *Ў Њ
°
3encoder/means/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@encoder/means/kernel*
valueB
 *Ў >*
dtype0*
_output_shapes
: 
е
=encoder/means/kernel/Initializer/random_uniform/RandomUniformRandomUniform5encoder/means/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@encoder/means/kernel*
dtype0*
_output_shapes
:	А

о
3encoder/means/kernel/Initializer/random_uniform/subSub3encoder/means/kernel/Initializer/random_uniform/max3encoder/means/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@encoder/means/kernel*
_output_shapes
: 
Б
3encoder/means/kernel/Initializer/random_uniform/mulMul=encoder/means/kernel/Initializer/random_uniform/RandomUniform3encoder/means/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@encoder/means/kernel*
_output_shapes
:	А

у
/encoder/means/kernel/Initializer/random_uniformAdd3encoder/means/kernel/Initializer/random_uniform/mul3encoder/means/kernel/Initializer/random_uniform/min*'
_class
loc:@encoder/means/kernel*
_output_shapes
:	А
*
T0
Ѓ
encoder/means/kernelVarHandleOp*'
_class
loc:@encoder/means/kernel*
dtype0*
_output_shapes
: *
shape:	А
*%
shared_nameencoder/means/kernel
y
5encoder/means/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/means/kernel*
_output_shapes
: 
ђ
encoder/means/kernel/AssignAssignVariableOpencoder/means/kernel/encoder/means/kernel/Initializer/random_uniform*
dtype0*'
_class
loc:@encoder/means/kernel
І
(encoder/means/kernel/Read/ReadVariableOpReadVariableOpencoder/means/kernel*
dtype0*
_output_shapes
:	А
*'
_class
loc:@encoder/means/kernel
Ш
$encoder/means/bias/Initializer/zerosConst*%
_class
loc:@encoder/means/bias*
valueB
*    *
dtype0*
_output_shapes
:

£
encoder/means/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*#
shared_nameencoder/means/bias*%
_class
loc:@encoder/means/bias
u
3encoder/means/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/means/bias*
_output_shapes
: 
Ы
encoder/means/bias/AssignAssignVariableOpencoder/means/bias$encoder/means/bias/Initializer/zeros*%
_class
loc:@encoder/means/bias*
dtype0
Ь
&encoder/means/bias/Read/ReadVariableOpReadVariableOpencoder/means/bias*%
_class
loc:@encoder/means/bias*
dtype0*
_output_shapes
:

y
#encoder/means/MatMul/ReadVariableOpReadVariableOpencoder/means/kernel*
dtype0*
_output_shapes
:	А

Ж
encoder/means/MatMulMatMulencoder/e5/Relu#encoder/means/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€

s
$encoder/means/BiasAdd/ReadVariableOpReadVariableOpencoder/means/bias*
dtype0*
_output_shapes
:

О
encoder/means/BiasAddBiasAddencoder/means/MatMul$encoder/means/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€

≥
7encoder/log_var/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@encoder/log_var/kernel*
valueB"   
   *
dtype0*
_output_shapes
:
•
5encoder/log_var/kernel/Initializer/random_uniform/minConst*)
_class
loc:@encoder/log_var/kernel*
valueB
 *Ў Њ*
dtype0*
_output_shapes
: 
•
5encoder/log_var/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@encoder/log_var/kernel*
valueB
 *Ў >*
dtype0*
_output_shapes
: 
л
?encoder/log_var/kernel/Initializer/random_uniform/RandomUniformRandomUniform7encoder/log_var/kernel/Initializer/random_uniform/shape*
_output_shapes
:	А
*
T0*)
_class
loc:@encoder/log_var/kernel*
dtype0
ц
5encoder/log_var/kernel/Initializer/random_uniform/subSub5encoder/log_var/kernel/Initializer/random_uniform/max5encoder/log_var/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@encoder/log_var/kernel*
_output_shapes
: 
Й
5encoder/log_var/kernel/Initializer/random_uniform/mulMul?encoder/log_var/kernel/Initializer/random_uniform/RandomUniform5encoder/log_var/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А
*
T0*)
_class
loc:@encoder/log_var/kernel
ы
1encoder/log_var/kernel/Initializer/random_uniformAdd5encoder/log_var/kernel/Initializer/random_uniform/mul5encoder/log_var/kernel/Initializer/random_uniform/min*)
_class
loc:@encoder/log_var/kernel*
_output_shapes
:	А
*
T0
і
encoder/log_var/kernelVarHandleOp*'
shared_nameencoder/log_var/kernel*)
_class
loc:@encoder/log_var/kernel*
dtype0*
_output_shapes
: *
shape:	А

}
7encoder/log_var/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/log_var/kernel*
_output_shapes
: 
і
encoder/log_var/kernel/AssignAssignVariableOpencoder/log_var/kernel1encoder/log_var/kernel/Initializer/random_uniform*)
_class
loc:@encoder/log_var/kernel*
dtype0
≠
*encoder/log_var/kernel/Read/ReadVariableOpReadVariableOpencoder/log_var/kernel*)
_class
loc:@encoder/log_var/kernel*
dtype0*
_output_shapes
:	А

Ь
&encoder/log_var/bias/Initializer/zerosConst*
_output_shapes
:
*'
_class
loc:@encoder/log_var/bias*
valueB
*    *
dtype0
©
encoder/log_var/biasVarHandleOp*
_output_shapes
: *
shape:
*%
shared_nameencoder/log_var/bias*'
_class
loc:@encoder/log_var/bias*
dtype0
y
5encoder/log_var/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpencoder/log_var/bias*
_output_shapes
: 
£
encoder/log_var/bias/AssignAssignVariableOpencoder/log_var/bias&encoder/log_var/bias/Initializer/zeros*'
_class
loc:@encoder/log_var/bias*
dtype0
Ґ
(encoder/log_var/bias/Read/ReadVariableOpReadVariableOpencoder/log_var/bias*'
_class
loc:@encoder/log_var/bias*
dtype0*
_output_shapes
:

}
%encoder/log_var/MatMul/ReadVariableOpReadVariableOpencoder/log_var/kernel*
_output_shapes
:	А
*
dtype0
К
encoder/log_var/MatMulMatMulencoder/e5/Relu%encoder/log_var/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€

w
&encoder/log_var/BiasAdd/ReadVariableOpReadVariableOpencoder/log_var/bias*
dtype0*
_output_shapes
:

Ф
encoder/log_var/BiasAddBiasAddencoder/log_var/MatMul&encoder/log_var/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€

N
	truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
h
truedivRealDivencoder/log_var/BiasAdd	truediv/y*
T0*'
_output_shapes
:€€€€€€€€€

E
ExpExptruediv*
T0*'
_output_shapes
:€€€€€€€€€

J
ShapeShapeencoder/means/BiasAdd*
T0*
_output_shapes
:
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
А
"random_normal/RandomStandardNormalRandomStandardNormalShape*'
_output_shapes
:€€€€€€€€€
*
T0*
dtype0
Д
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*'
_output_shapes
:€€€€€€€€€

m
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*'
_output_shapes
:€€€€€€€€€

P
mulMulExprandom_normal*'
_output_shapes
:€€€€€€€€€
*
T0
l
sampled_latent_variableAddencoder/means/BiasAddmul*
T0*'
_output_shapes
:€€€€€€€€€

ѓ
5decoder/dense/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@decoder/dense/kernel*
valueB"
      *
dtype0*
_output_shapes
:
°
3decoder/dense/kernel/Initializer/random_uniform/minConst*'
_class
loc:@decoder/dense/kernel*
valueB
 *Ў Њ*
dtype0*
_output_shapes
: 
°
3decoder/dense/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@decoder/dense/kernel*
valueB
 *Ў >*
dtype0*
_output_shapes
: 
е
=decoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5decoder/dense/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@decoder/dense/kernel*
dtype0*
_output_shapes
:	
А
о
3decoder/dense/kernel/Initializer/random_uniform/subSub3decoder/dense/kernel/Initializer/random_uniform/max3decoder/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@decoder/dense/kernel
Б
3decoder/dense/kernel/Initializer/random_uniform/mulMul=decoder/dense/kernel/Initializer/random_uniform/RandomUniform3decoder/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@decoder/dense/kernel*
_output_shapes
:	
А
у
/decoder/dense/kernel/Initializer/random_uniformAdd3decoder/dense/kernel/Initializer/random_uniform/mul3decoder/dense/kernel/Initializer/random_uniform/min*'
_class
loc:@decoder/dense/kernel*
_output_shapes
:	
А*
T0
Ѓ
decoder/dense/kernelVarHandleOp*
shape:	
А*%
shared_namedecoder/dense/kernel*'
_class
loc:@decoder/dense/kernel*
dtype0*
_output_shapes
: 
y
5decoder/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/dense/kernel*
_output_shapes
: 
ђ
decoder/dense/kernel/AssignAssignVariableOpdecoder/dense/kernel/decoder/dense/kernel/Initializer/random_uniform*'
_class
loc:@decoder/dense/kernel*
dtype0
І
(decoder/dense/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense/kernel*'
_class
loc:@decoder/dense/kernel*
dtype0*
_output_shapes
:	
А
Ъ
$decoder/dense/bias/Initializer/zerosConst*%
_class
loc:@decoder/dense/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
§
decoder/dense/biasVarHandleOp*
shape:А*#
shared_namedecoder/dense/bias*%
_class
loc:@decoder/dense/bias*
dtype0*
_output_shapes
: 
u
3decoder/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/dense/bias*
_output_shapes
: 
Ы
decoder/dense/bias/AssignAssignVariableOpdecoder/dense/bias$decoder/dense/bias/Initializer/zeros*%
_class
loc:@decoder/dense/bias*
dtype0
Э
&decoder/dense/bias/Read/ReadVariableOpReadVariableOpdecoder/dense/bias*%
_class
loc:@decoder/dense/bias*
dtype0*
_output_shapes	
:А
y
#decoder/dense/MatMul/ReadVariableOpReadVariableOpdecoder/dense/kernel*
dtype0*
_output_shapes
:	
А
П
decoder/dense/MatMulMatMulsampled_latent_variable#decoder/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
t
$decoder/dense/BiasAdd/ReadVariableOpReadVariableOpdecoder/dense/bias*
dtype0*
_output_shapes	
:А
П
decoder/dense/BiasAddBiasAdddecoder/dense/MatMul$decoder/dense/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
d
decoder/dense/ReluReludecoder/dense/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
≥
7decoder/dense_1/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@decoder/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
•
5decoder/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *)
_class
loc:@decoder/dense_1/kernel*
valueB
 *М7Мљ
•
5decoder/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *)
_class
loc:@decoder/dense_1/kernel*
valueB
 *М7М=
м
?decoder/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7decoder/dense_1/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@decoder/dense_1/kernel*
dtype0* 
_output_shapes
:
АА
ц
5decoder/dense_1/kernel/Initializer/random_uniform/subSub5decoder/dense_1/kernel/Initializer/random_uniform/max5decoder/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@decoder/dense_1/kernel
К
5decoder/dense_1/kernel/Initializer/random_uniform/mulMul?decoder/dense_1/kernel/Initializer/random_uniform/RandomUniform5decoder/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@decoder/dense_1/kernel* 
_output_shapes
:
АА
ь
1decoder/dense_1/kernel/Initializer/random_uniformAdd5decoder/dense_1/kernel/Initializer/random_uniform/mul5decoder/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@decoder/dense_1/kernel* 
_output_shapes
:
АА
µ
decoder/dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
АА*'
shared_namedecoder/dense_1/kernel*)
_class
loc:@decoder/dense_1/kernel
}
7decoder/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/dense_1/kernel*
_output_shapes
: 
і
decoder/dense_1/kernel/AssignAssignVariableOpdecoder/dense_1/kernel1decoder/dense_1/kernel/Initializer/random_uniform*)
_class
loc:@decoder/dense_1/kernel*
dtype0
Ѓ
*decoder/dense_1/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_1/kernel*)
_class
loc:@decoder/dense_1/kernel*
dtype0* 
_output_shapes
:
АА
™
6decoder/dense_1/bias/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@decoder/dense_1/bias*
valueB:А*
dtype0*
_output_shapes
:
Ъ
,decoder/dense_1/bias/Initializer/zeros/ConstConst*'
_class
loc:@decoder/dense_1/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
г
&decoder/dense_1/bias/Initializer/zerosFill6decoder/dense_1/bias/Initializer/zeros/shape_as_tensor,decoder/dense_1/bias/Initializer/zeros/Const*
_output_shapes	
:А*
T0*'
_class
loc:@decoder/dense_1/bias
™
decoder/dense_1/biasVarHandleOp*'
_class
loc:@decoder/dense_1/bias*
dtype0*
_output_shapes
: *
shape:А*%
shared_namedecoder/dense_1/bias
y
5decoder/dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/dense_1/bias*
_output_shapes
: 
£
decoder/dense_1/bias/AssignAssignVariableOpdecoder/dense_1/bias&decoder/dense_1/bias/Initializer/zeros*'
_class
loc:@decoder/dense_1/bias*
dtype0
£
(decoder/dense_1/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_1/bias*'
_class
loc:@decoder/dense_1/bias*
dtype0*
_output_shapes	
:А
~
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOpdecoder/dense_1/kernel* 
_output_shapes
:
АА*
dtype0
О
decoder/dense_1/MatMulMatMuldecoder/dense/Relu%decoder/dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
x
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/dense_1/bias*
dtype0*
_output_shapes	
:А
Х
decoder/dense_1/BiasAddBiasAdddecoder/dense_1/MatMul&decoder/dense_1/BiasAdd/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
T0
h
decoder/dense_1/ReluReludecoder/dense_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
n
decoder/Reshape/shapeConst*
_output_shapes
:*%
valueB"€€€€      @   *
dtype0
Б
decoder/ReshapeReshapedecoder/dense_1/Reludecoder/Reshape/shape*/
_output_shapes
:€€€€€€€€€@*
T0
Ќ
@decoder/conv2d_transpose/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*%
valueB"      @   @   *
dtype0
Ј
>decoder/conv2d_transpose/kernel/Initializer/random_uniform/minConst*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
valueB
 *„≥]љ*
dtype0*
_output_shapes
: 
Ј
>decoder/conv2d_transpose/kernel/Initializer/random_uniform/maxConst*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
valueB
 *„≥]=*
dtype0*
_output_shapes
: 
Н
Hdecoder/conv2d_transpose/kernel/Initializer/random_uniform/RandomUniformRandomUniform@decoder/conv2d_transpose/kernel/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0*&
_output_shapes
:@@
Ъ
>decoder/conv2d_transpose/kernel/Initializer/random_uniform/subSub>decoder/conv2d_transpose/kernel/Initializer/random_uniform/max>decoder/conv2d_transpose/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
_output_shapes
: 
і
>decoder/conv2d_transpose/kernel/Initializer/random_uniform/mulMulHdecoder/conv2d_transpose/kernel/Initializer/random_uniform/RandomUniform>decoder/conv2d_transpose/kernel/Initializer/random_uniform/sub*&
_output_shapes
:@@*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel
¶
:decoder/conv2d_transpose/kernel/Initializer/random_uniformAdd>decoder/conv2d_transpose/kernel/Initializer/random_uniform/mul>decoder/conv2d_transpose/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*&
_output_shapes
:@@
÷
decoder/conv2d_transpose/kernelVarHandleOp*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0*
_output_shapes
: *
shape:@@*0
shared_name!decoder/conv2d_transpose/kernel
П
@decoder/conv2d_transpose/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose/kernel*
_output_shapes
: 
Ў
&decoder/conv2d_transpose/kernel/AssignAssignVariableOpdecoder/conv2d_transpose/kernel:decoder/conv2d_transpose/kernel/Initializer/random_uniform*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0
ѕ
3decoder/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/kernel*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0*&
_output_shapes
:@@
Ѓ
/decoder/conv2d_transpose/bias/Initializer/zerosConst*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
valueB@*    *
dtype0*
_output_shapes
:@
ƒ
decoder/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
shape:@*.
shared_namedecoder/conv2d_transpose/bias*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0
Л
>decoder/conv2d_transpose/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose/bias*
_output_shapes
: 
«
$decoder/conv2d_transpose/bias/AssignAssignVariableOpdecoder/conv2d_transpose/bias/decoder/conv2d_transpose/bias/Initializer/zeros*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0
љ
1decoder/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/bias*
dtype0*
_output_shapes
:@*0
_class&
$"loc:@decoder/conv2d_transpose/bias
]
decoder/conv2d_transpose/ShapeShapedecoder/Reshape*
T0*
_output_shapes
:
v
,decoder/conv2d_transpose/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.decoder/conv2d_transpose/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.decoder/conv2d_transpose/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
™
&decoder/conv2d_transpose/strided_sliceStridedSlicedecoder/conv2d_transpose/Shape,decoder/conv2d_transpose/strided_slice/stack.decoder/conv2d_transpose/strided_slice/stack_1.decoder/conv2d_transpose/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
x
.decoder/conv2d_transpose/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
≤
(decoder/conv2d_transpose/strided_slice_1StridedSlicedecoder/conv2d_transpose/Shape.decoder/conv2d_transpose/strided_slice_1/stack0decoder/conv2d_transpose/strided_slice_1/stack_10decoder/conv2d_transpose/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
x
.decoder/conv2d_transpose/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
z
0decoder/conv2d_transpose/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
≤
(decoder/conv2d_transpose/strided_slice_2StridedSlicedecoder/conv2d_transpose/Shape.decoder/conv2d_transpose/strided_slice_2/stack0decoder/conv2d_transpose/strided_slice_2/stack_10decoder/conv2d_transpose/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
`
decoder/conv2d_transpose/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
О
decoder/conv2d_transpose/mulMul(decoder/conv2d_transpose/strided_slice_1decoder/conv2d_transpose/mul/y*
T0*
_output_shapes
: 
b
 decoder/conv2d_transpose/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Т
decoder/conv2d_transpose/mul_1Mul(decoder/conv2d_transpose/strided_slice_2 decoder/conv2d_transpose/mul_1/y*
T0*
_output_shapes
: 
b
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
value	B :@*
dtype0
№
decoder/conv2d_transpose/stackPack&decoder/conv2d_transpose/strided_slicedecoder/conv2d_transpose/muldecoder/conv2d_transpose/mul_1 decoder/conv2d_transpose/stack/3*
T0*
N*
_output_shapes
:
x
.decoder/conv2d_transpose/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB: 
z
0decoder/conv2d_transpose/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
≤
(decoder/conv2d_transpose/strided_slice_3StridedSlicedecoder/conv2d_transpose/stack.decoder/conv2d_transpose/strided_slice_3/stack0decoder/conv2d_transpose/strided_slice_3/stack_10decoder/conv2d_transpose/strided_slice_3/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
†
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/kernel*
dtype0*&
_output_shapes
:@@
Н
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInputdecoder/conv2d_transpose/stack8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpdecoder/Reshape*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€@
Й
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/bias*
dtype0*
_output_shapes
:@
Ѕ
 decoder/conv2d_transpose/BiasAddBiasAdd)decoder/conv2d_transpose/conv2d_transpose/decoder/conv2d_transpose/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€@*
T0
Б
decoder/conv2d_transpose/ReluRelu decoder/conv2d_transpose/BiasAdd*/
_output_shapes
:€€€€€€€€€@*
T0
—
Bdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/shapeConst*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*%
valueB"          @   *
dtype0*
_output_shapes
:
ї
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/minConst*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
valueB
 *  Аљ*
dtype0*
_output_shapes
: 
ї
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/maxConst*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
valueB
 *  А=*
dtype0*
_output_shapes
: 
У
Jdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformBdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/shape*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*&
_output_shapes
: @
Ґ
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/subSub@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/max@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
_output_shapes
: 
Љ
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/mulMulJdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniform@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*&
_output_shapes
: @
Ѓ
<decoder/conv2d_transpose_1/kernel/Initializer/random_uniformAdd@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/mul@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/min*&
_output_shapes
: @*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel
№
!decoder/conv2d_transpose_1/kernelVarHandleOp*2
shared_name#!decoder/conv2d_transpose_1/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*
_output_shapes
: *
shape: @
У
Bdecoder/conv2d_transpose_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!decoder/conv2d_transpose_1/kernel*
_output_shapes
: 
а
(decoder/conv2d_transpose_1/kernel/AssignAssignVariableOp!decoder/conv2d_transpose_1/kernel<decoder/conv2d_transpose_1/kernel/Initializer/random_uniform*
dtype0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel
’
5decoder/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_1/kernel*&
_output_shapes
: @*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0
≤
1decoder/conv2d_transpose_1/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
valueB *    *
dtype0*
_output_shapes
: 
 
decoder/conv2d_transpose_1/biasVarHandleOp*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
dtype0*
_output_shapes
: *
shape: *0
shared_name!decoder/conv2d_transpose_1/bias
П
@decoder/conv2d_transpose_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose_1/bias*
_output_shapes
: 
ѕ
&decoder/conv2d_transpose_1/bias/AssignAssignVariableOpdecoder/conv2d_transpose_1/bias1decoder/conv2d_transpose_1/bias/Initializer/zeros*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
dtype0
√
3decoder/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_1/bias*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
dtype0*
_output_shapes
: 
m
 decoder/conv2d_transpose_1/ShapeShapedecoder/conv2d_transpose/Relu*
T0*
_output_shapes
:
x
.decoder/conv2d_transpose_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
(decoder/conv2d_transpose_1/strided_sliceStridedSlice decoder/conv2d_transpose_1/Shape.decoder/conv2d_transpose_1/strided_slice/stack0decoder/conv2d_transpose_1/strided_slice/stack_10decoder/conv2d_transpose_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
z
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice decoder/conv2d_transpose_1/Shape0decoder/conv2d_transpose_1/strided_slice_1/stack2decoder/conv2d_transpose_1/strided_slice_1/stack_12decoder/conv2d_transpose_1/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
z
0decoder/conv2d_transpose_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Љ
*decoder/conv2d_transpose_1/strided_slice_2StridedSlice decoder/conv2d_transpose_1/Shape0decoder/conv2d_transpose_1/strided_slice_2/stack2decoder/conv2d_transpose_1/strided_slice_2/stack_12decoder/conv2d_transpose_1/strided_slice_2/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
b
 decoder/conv2d_transpose_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ф
decoder/conv2d_transpose_1/mulMul*decoder/conv2d_transpose_1/strided_slice_1 decoder/conv2d_transpose_1/mul/y*
T0*
_output_shapes
: 
d
"decoder/conv2d_transpose_1/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
 decoder/conv2d_transpose_1/mul_1Mul*decoder/conv2d_transpose_1/strided_slice_2"decoder/conv2d_transpose_1/mul_1/y*
_output_shapes
: *
T0
d
"decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
value	B : *
dtype0
ж
 decoder/conv2d_transpose_1/stackPack(decoder/conv2d_transpose_1/strided_slicedecoder/conv2d_transpose_1/mul decoder/conv2d_transpose_1/mul_1"decoder/conv2d_transpose_1/stack/3*
N*
_output_shapes
:*
T0
z
0decoder/conv2d_transpose_1/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
|
2decoder/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Љ
*decoder/conv2d_transpose_1/strided_slice_3StridedSlice decoder/conv2d_transpose_1/stack0decoder/conv2d_transpose_1/strided_slice_3/stack2decoder/conv2d_transpose_1/strided_slice_3/stack_12decoder/conv2d_transpose_1/strided_slice_3/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
§
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_1/kernel*
dtype0*&
_output_shapes
: @
°
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput decoder/conv2d_transpose_1/stack:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpdecoder/conv2d_transpose/Relu*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€ 
Н
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_1/bias*
dtype0*
_output_shapes
: 
«
"decoder/conv2d_transpose_1/BiasAddBiasAdd+decoder/conv2d_transpose_1/conv2d_transpose1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 
Е
decoder/conv2d_transpose_1/ReluRelu"decoder/conv2d_transpose_1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€ 
—
Bdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/shapeConst*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*%
valueB"              *
dtype0*
_output_shapes
:
ї
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/minConst*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
valueB
 *qƒЬљ*
dtype0*
_output_shapes
: 
ї
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/maxConst*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
valueB
 *qƒЬ=*
dtype0*
_output_shapes
: 
У
Jdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformBdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:  *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel
Ґ
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/subSub@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/max@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel
Љ
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/mulMulJdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniform@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*&
_output_shapes
:  
Ѓ
<decoder/conv2d_transpose_2/kernel/Initializer/random_uniformAdd@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/mul@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*&
_output_shapes
:  
№
!decoder/conv2d_transpose_2/kernelVarHandleOp*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0*
_output_shapes
: *
shape:  *2
shared_name#!decoder/conv2d_transpose_2/kernel
У
Bdecoder/conv2d_transpose_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!decoder/conv2d_transpose_2/kernel*
_output_shapes
: 
а
(decoder/conv2d_transpose_2/kernel/AssignAssignVariableOp!decoder/conv2d_transpose_2/kernel<decoder/conv2d_transpose_2/kernel/Initializer/random_uniform*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0
’
5decoder/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_2/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0*&
_output_shapes
:  
≤
1decoder/conv2d_transpose_2/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
valueB *    *
dtype0*
_output_shapes
: 
 
decoder/conv2d_transpose_2/biasVarHandleOp*0
shared_name!decoder/conv2d_transpose_2/bias*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0*
_output_shapes
: *
shape: 
П
@decoder/conv2d_transpose_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose_2/bias*
_output_shapes
: 
ѕ
&decoder/conv2d_transpose_2/bias/AssignAssignVariableOpdecoder/conv2d_transpose_2/bias1decoder/conv2d_transpose_2/bias/Initializer/zeros*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0
√
3decoder/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_2/bias*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0*
_output_shapes
: 
o
 decoder/conv2d_transpose_2/ShapeShapedecoder/conv2d_transpose_1/Relu*
_output_shapes
:*
T0
x
.decoder/conv2d_transpose_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
(decoder/conv2d_transpose_2/strided_sliceStridedSlice decoder/conv2d_transpose_2/Shape.decoder/conv2d_transpose_2/strided_slice/stack0decoder/conv2d_transpose_2/strided_slice/stack_10decoder/conv2d_transpose_2/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
z
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice decoder/conv2d_transpose_2/Shape0decoder/conv2d_transpose_2/strided_slice_1/stack2decoder/conv2d_transpose_2/strided_slice_1/stack_12decoder/conv2d_transpose_2/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
z
0decoder/conv2d_transpose_2/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_2/strided_slice_2StridedSlice decoder/conv2d_transpose_2/Shape0decoder/conv2d_transpose_2/strided_slice_2/stack2decoder/conv2d_transpose_2/strided_slice_2/stack_12decoder/conv2d_transpose_2/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
b
 decoder/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
value	B :*
dtype0
Ф
decoder/conv2d_transpose_2/mulMul*decoder/conv2d_transpose_2/strided_slice_1 decoder/conv2d_transpose_2/mul/y*
T0*
_output_shapes
: 
d
"decoder/conv2d_transpose_2/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
 decoder/conv2d_transpose_2/mul_1Mul*decoder/conv2d_transpose_2/strided_slice_2"decoder/conv2d_transpose_2/mul_1/y*
T0*
_output_shapes
: 
d
"decoder/conv2d_transpose_2/stack/3Const*
dtype0*
_output_shapes
: *
value	B : 
ж
 decoder/conv2d_transpose_2/stackPack(decoder/conv2d_transpose_2/strided_slicedecoder/conv2d_transpose_2/mul decoder/conv2d_transpose_2/mul_1"decoder/conv2d_transpose_2/stack/3*
T0*
N*
_output_shapes
:
z
0decoder/conv2d_transpose_2/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_2/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_2/strided_slice_3StridedSlice decoder/conv2d_transpose_2/stack0decoder/conv2d_transpose_2/strided_slice_3/stack2decoder/conv2d_transpose_2/strided_slice_3/stack_12decoder/conv2d_transpose_2/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
§
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_2/kernel*
dtype0*&
_output_shapes
:  
£
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput decoder/conv2d_transpose_2/stack:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpdecoder/conv2d_transpose_1/Relu*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€   
Н
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_2/bias*
dtype0*
_output_shapes
: 
«
"decoder/conv2d_transpose_2/BiasAddBiasAdd+decoder/conv2d_transpose_2/conv2d_transpose1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   
Е
decoder/conv2d_transpose_2/ReluRelu"decoder/conv2d_transpose_2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€   
—
Bdecoder/conv2d_transpose_3/kernel/Initializer/random_uniform/shapeConst*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*%
valueB"             *
dtype0*
_output_shapes
:
ї
@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/minConst*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
valueB
 *чь”љ*
dtype0*
_output_shapes
: 
ї
@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/maxConst*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
valueB
 *чь”=*
dtype0*
_output_shapes
: 
У
Jdecoder/conv2d_transpose_3/kernel/Initializer/random_uniform/RandomUniformRandomUniformBdecoder/conv2d_transpose_3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel
Ґ
@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/subSub@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/max@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
_output_shapes
: 
Љ
@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/mulMulJdecoder/conv2d_transpose_3/kernel/Initializer/random_uniform/RandomUniform@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*&
_output_shapes
: 
Ѓ
<decoder/conv2d_transpose_3/kernel/Initializer/random_uniformAdd@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/mul@decoder/conv2d_transpose_3/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*&
_output_shapes
: 
№
!decoder/conv2d_transpose_3/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: *2
shared_name#!decoder/conv2d_transpose_3/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel
У
Bdecoder/conv2d_transpose_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!decoder/conv2d_transpose_3/kernel*
_output_shapes
: 
а
(decoder/conv2d_transpose_3/kernel/AssignAssignVariableOp!decoder/conv2d_transpose_3/kernel<decoder/conv2d_transpose_3/kernel/Initializer/random_uniform*
dtype0*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel
’
5decoder/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_3/kernel*4
_class*
(&loc:@decoder/conv2d_transpose_3/kernel*
dtype0*&
_output_shapes
: 
≤
1decoder/conv2d_transpose_3/bias/Initializer/zerosConst*
_output_shapes
:*2
_class(
&$loc:@decoder/conv2d_transpose_3/bias*
valueB*    *
dtype0
 
decoder/conv2d_transpose_3/biasVarHandleOp*0
shared_name!decoder/conv2d_transpose_3/bias*2
_class(
&$loc:@decoder/conv2d_transpose_3/bias*
dtype0*
_output_shapes
: *
shape:
П
@decoder/conv2d_transpose_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/conv2d_transpose_3/bias*
_output_shapes
: 
ѕ
&decoder/conv2d_transpose_3/bias/AssignAssignVariableOpdecoder/conv2d_transpose_3/bias1decoder/conv2d_transpose_3/bias/Initializer/zeros*2
_class(
&$loc:@decoder/conv2d_transpose_3/bias*
dtype0
√
3decoder/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_3/bias*2
_class(
&$loc:@decoder/conv2d_transpose_3/bias*
dtype0*
_output_shapes
:
o
 decoder/conv2d_transpose_3/ShapeShapedecoder/conv2d_transpose_2/Relu*
_output_shapes
:*
T0
x
.decoder/conv2d_transpose_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
z
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
(decoder/conv2d_transpose_3/strided_sliceStridedSlice decoder/conv2d_transpose_3/Shape.decoder/conv2d_transpose_3/strided_slice/stack0decoder/conv2d_transpose_3/strided_slice/stack_10decoder/conv2d_transpose_3/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
z
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
|
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Љ
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice decoder/conv2d_transpose_3/Shape0decoder/conv2d_transpose_3/strided_slice_1/stack2decoder/conv2d_transpose_3/strided_slice_1/stack_12decoder/conv2d_transpose_3/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
z
0decoder/conv2d_transpose_3/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_3/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_3/strided_slice_2StridedSlice decoder/conv2d_transpose_3/Shape0decoder/conv2d_transpose_3/strided_slice_2/stack2decoder/conv2d_transpose_3/strided_slice_2/stack_12decoder/conv2d_transpose_3/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
b
 decoder/conv2d_transpose_3/mul/yConst*
dtype0*
_output_shapes
: *
value	B :
Ф
decoder/conv2d_transpose_3/mulMul*decoder/conv2d_transpose_3/strided_slice_1 decoder/conv2d_transpose_3/mul/y*
_output_shapes
: *
T0
d
"decoder/conv2d_transpose_3/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
 decoder/conv2d_transpose_3/mul_1Mul*decoder/conv2d_transpose_3/strided_slice_2"decoder/conv2d_transpose_3/mul_1/y*
T0*
_output_shapes
: 
d
"decoder/conv2d_transpose_3/stack/3Const*
value	B :*
dtype0*
_output_shapes
: 
ж
 decoder/conv2d_transpose_3/stackPack(decoder/conv2d_transpose_3/strided_slicedecoder/conv2d_transpose_3/mul decoder/conv2d_transpose_3/mul_1"decoder/conv2d_transpose_3/stack/3*
N*
_output_shapes
:*
T0
z
0decoder/conv2d_transpose_3/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_3/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder/conv2d_transpose_3/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder/conv2d_transpose_3/strided_slice_3StridedSlice decoder/conv2d_transpose_3/stack0decoder/conv2d_transpose_3/strided_slice_3/stack2decoder/conv2d_transpose_3/strided_slice_3/stack_12decoder/conv2d_transpose_3/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
§
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_3/kernel*&
_output_shapes
: *
dtype0
£
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput decoder/conv2d_transpose_3/stack:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpdecoder/conv2d_transpose_2/Relu*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€@@
Н
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_3/bias*
dtype0*
_output_shapes
:
«
"decoder/conv2d_transpose_3/BiasAddBiasAdd+decoder/conv2d_transpose_3/conv2d_transpose1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@
p
decoder/Reshape_1/shapeConst*%
valueB"€€€€@   @      *
dtype0*
_output_shapes
:
У
decoder/Reshape_1Reshape"decoder/conv2d_transpose_3/BiasAdddecoder/Reshape_1/shape*
T0*/
_output_shapes
:€€€€€€€€€@@
p
Placeholder_1Placeholder*
shape:€€€€€€€€€
*
dtype0*'
_output_shapes
:€€€€€€€€€

{
%decoder_1/dense/MatMul/ReadVariableOpReadVariableOpdecoder/dense/kernel*
dtype0*
_output_shapes
:	
А
Й
decoder_1/dense/MatMulMatMulPlaceholder_1%decoder_1/dense/MatMul/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
T0
v
&decoder_1/dense/BiasAdd/ReadVariableOpReadVariableOpdecoder/dense/bias*
dtype0*
_output_shapes	
:А
Х
decoder_1/dense/BiasAddBiasAdddecoder_1/dense/MatMul&decoder_1/dense/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
h
decoder_1/dense/ReluReludecoder_1/dense/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
А
'decoder_1/dense_1/MatMul/ReadVariableOpReadVariableOpdecoder/dense_1/kernel*
dtype0* 
_output_shapes
:
АА
Ф
decoder_1/dense_1/MatMulMatMuldecoder_1/dense/Relu'decoder_1/dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А
z
(decoder_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/dense_1/bias*
dtype0*
_output_shapes	
:А
Ы
decoder_1/dense_1/BiasAddBiasAdddecoder_1/dense_1/MatMul(decoder_1/dense_1/BiasAdd/ReadVariableOp*(
_output_shapes
:€€€€€€€€€А*
T0
l
decoder_1/dense_1/ReluReludecoder_1/dense_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
p
decoder_1/Reshape/shapeConst*
_output_shapes
:*%
valueB"€€€€      @   *
dtype0
З
decoder_1/ReshapeReshapedecoder_1/dense_1/Reludecoder_1/Reshape/shape*
T0*/
_output_shapes
:€€€€€€€€€@
a
 decoder_1/conv2d_transpose/ShapeShapedecoder_1/Reshape*
T0*
_output_shapes
:
x
.decoder_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
z
0decoder_1/conv2d_transpose/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder_1/conv2d_transpose/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
і
(decoder_1/conv2d_transpose/strided_sliceStridedSlice decoder_1/conv2d_transpose/Shape.decoder_1/conv2d_transpose/strided_slice/stack0decoder_1/conv2d_transpose/strided_slice/stack_10decoder_1/conv2d_transpose/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
z
0decoder_1/conv2d_transpose/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
|
2decoder_1/conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder_1/conv2d_transpose/strided_slice_1StridedSlice decoder_1/conv2d_transpose/Shape0decoder_1/conv2d_transpose/strided_slice_1/stack2decoder_1/conv2d_transpose/strided_slice_1/stack_12decoder_1/conv2d_transpose/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
z
0decoder_1/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
valueB:*
dtype0
|
2decoder_1/conv2d_transpose/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
|
2decoder_1/conv2d_transpose/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Љ
*decoder_1/conv2d_transpose/strided_slice_2StridedSlice decoder_1/conv2d_transpose/Shape0decoder_1/conv2d_transpose/strided_slice_2/stack2decoder_1/conv2d_transpose/strided_slice_2/stack_12decoder_1/conv2d_transpose/strided_slice_2/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
b
 decoder_1/conv2d_transpose/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ф
decoder_1/conv2d_transpose/mulMul*decoder_1/conv2d_transpose/strided_slice_1 decoder_1/conv2d_transpose/mul/y*
T0*
_output_shapes
: 
d
"decoder_1/conv2d_transpose/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
 decoder_1/conv2d_transpose/mul_1Mul*decoder_1/conv2d_transpose/strided_slice_2"decoder_1/conv2d_transpose/mul_1/y*
T0*
_output_shapes
: 
d
"decoder_1/conv2d_transpose/stack/3Const*
value	B :@*
dtype0*
_output_shapes
: 
ж
 decoder_1/conv2d_transpose/stackPack(decoder_1/conv2d_transpose/strided_slicedecoder_1/conv2d_transpose/mul decoder_1/conv2d_transpose/mul_1"decoder_1/conv2d_transpose/stack/3*
T0*
N*
_output_shapes
:
z
0decoder_1/conv2d_transpose/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Љ
*decoder_1/conv2d_transpose/strided_slice_3StridedSlice decoder_1/conv2d_transpose/stack0decoder_1/conv2d_transpose/strided_slice_3/stack2decoder_1/conv2d_transpose/strided_slice_3/stack_12decoder_1/conv2d_transpose/strided_slice_3/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
Ґ
:decoder_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/kernel*
dtype0*&
_output_shapes
:@@
Х
+decoder_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput decoder_1/conv2d_transpose/stack:decoder_1/conv2d_transpose/conv2d_transpose/ReadVariableOpdecoder_1/Reshape*
strides
*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
T0
Л
1decoder_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose/bias*
dtype0*
_output_shapes
:@
«
"decoder_1/conv2d_transpose/BiasAddBiasAdd+decoder_1/conv2d_transpose/conv2d_transpose1decoder_1/conv2d_transpose/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@
Е
decoder_1/conv2d_transpose/ReluRelu"decoder_1/conv2d_transpose/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
q
"decoder_1/conv2d_transpose_1/ShapeShapedecoder_1/conv2d_transpose/Relu*
T0*
_output_shapes
:
z
0decoder_1/conv2d_transpose_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Њ
*decoder_1/conv2d_transpose_1/strided_sliceStridedSlice"decoder_1/conv2d_transpose_1/Shape0decoder_1/conv2d_transpose_1/strided_slice/stack2decoder_1/conv2d_transpose_1/strided_slice/stack_12decoder_1/conv2d_transpose_1/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
|
2decoder_1/conv2d_transpose_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
~
4decoder_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
~
4decoder_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_1/strided_slice_1StridedSlice"decoder_1/conv2d_transpose_1/Shape2decoder_1/conv2d_transpose_1/strided_slice_1/stack4decoder_1/conv2d_transpose_1/strided_slice_1/stack_14decoder_1/conv2d_transpose_1/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
|
2decoder_1/conv2d_transpose_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_1/strided_slice_2StridedSlice"decoder_1/conv2d_transpose_1/Shape2decoder_1/conv2d_transpose_1/strided_slice_2/stack4decoder_1/conv2d_transpose_1/strided_slice_2/stack_14decoder_1/conv2d_transpose_1/strided_slice_2/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
d
"decoder_1/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
value	B :*
dtype0
Ъ
 decoder_1/conv2d_transpose_1/mulMul,decoder_1/conv2d_transpose_1/strided_slice_1"decoder_1/conv2d_transpose_1/mul/y*
_output_shapes
: *
T0
f
$decoder_1/conv2d_transpose_1/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ю
"decoder_1/conv2d_transpose_1/mul_1Mul,decoder_1/conv2d_transpose_1/strided_slice_2$decoder_1/conv2d_transpose_1/mul_1/y*
T0*
_output_shapes
: 
f
$decoder_1/conv2d_transpose_1/stack/3Const*
value	B : *
dtype0*
_output_shapes
: 
р
"decoder_1/conv2d_transpose_1/stackPack*decoder_1/conv2d_transpose_1/strided_slice decoder_1/conv2d_transpose_1/mul"decoder_1/conv2d_transpose_1/mul_1$decoder_1/conv2d_transpose_1/stack/3*
T0*
N*
_output_shapes
:
|
2decoder_1/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
valueB: *
dtype0
~
4decoder_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
~
4decoder_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_1/strided_slice_3StridedSlice"decoder_1/conv2d_transpose_1/stack2decoder_1/conv2d_transpose_1/strided_slice_3/stack4decoder_1/conv2d_transpose_1/strided_slice_3/stack_14decoder_1/conv2d_transpose_1/strided_slice_3/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
¶
<decoder_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_1/kernel*
dtype0*&
_output_shapes
: @
©
-decoder_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput"decoder_1/conv2d_transpose_1/stack<decoder_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpdecoder_1/conv2d_transpose/Relu*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€ 
П
3decoder_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_1/bias*
dtype0*
_output_shapes
: 
Ќ
$decoder_1/conv2d_transpose_1/BiasAddBiasAdd-decoder_1/conv2d_transpose_1/conv2d_transpose3decoder_1/conv2d_transpose_1/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€ *
T0
Й
!decoder_1/conv2d_transpose_1/ReluRelu$decoder_1/conv2d_transpose_1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€ 
s
"decoder_1/conv2d_transpose_2/ShapeShape!decoder_1/conv2d_transpose_1/Relu*
T0*
_output_shapes
:
z
0decoder_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
|
2decoder_1/conv2d_transpose_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
|
2decoder_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Њ
*decoder_1/conv2d_transpose_2/strided_sliceStridedSlice"decoder_1/conv2d_transpose_2/Shape0decoder_1/conv2d_transpose_2/strided_slice/stack2decoder_1/conv2d_transpose_2/strided_slice/stack_12decoder_1/conv2d_transpose_2/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
|
2decoder_1/conv2d_transpose_2/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
~
4decoder_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_2/strided_slice_1StridedSlice"decoder_1/conv2d_transpose_2/Shape2decoder_1/conv2d_transpose_2/strided_slice_1/stack4decoder_1/conv2d_transpose_2/strided_slice_1/stack_14decoder_1/conv2d_transpose_2/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
|
2decoder_1/conv2d_transpose_2/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_2/strided_slice_2StridedSlice"decoder_1/conv2d_transpose_2/Shape2decoder_1/conv2d_transpose_2/strided_slice_2/stack4decoder_1/conv2d_transpose_2/strided_slice_2/stack_14decoder_1/conv2d_transpose_2/strided_slice_2/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
d
"decoder_1/conv2d_transpose_2/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ъ
 decoder_1/conv2d_transpose_2/mulMul,decoder_1/conv2d_transpose_2/strided_slice_1"decoder_1/conv2d_transpose_2/mul/y*
_output_shapes
: *
T0
f
$decoder_1/conv2d_transpose_2/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ю
"decoder_1/conv2d_transpose_2/mul_1Mul,decoder_1/conv2d_transpose_2/strided_slice_2$decoder_1/conv2d_transpose_2/mul_1/y*
T0*
_output_shapes
: 
f
$decoder_1/conv2d_transpose_2/stack/3Const*
value	B : *
dtype0*
_output_shapes
: 
р
"decoder_1/conv2d_transpose_2/stackPack*decoder_1/conv2d_transpose_2/strided_slice decoder_1/conv2d_transpose_2/mul"decoder_1/conv2d_transpose_2/mul_1$decoder_1/conv2d_transpose_2/stack/3*
T0*
N*
_output_shapes
:
|
2decoder_1/conv2d_transpose_2/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
∆
,decoder_1/conv2d_transpose_2/strided_slice_3StridedSlice"decoder_1/conv2d_transpose_2/stack2decoder_1/conv2d_transpose_2/strided_slice_3/stack4decoder_1/conv2d_transpose_2/strided_slice_3/stack_14decoder_1/conv2d_transpose_2/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
¶
<decoder_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_2/kernel*
dtype0*&
_output_shapes
:  
Ђ
-decoder_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput"decoder_1/conv2d_transpose_2/stack<decoder_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp!decoder_1/conv2d_transpose_1/Relu*
paddingSAME*
T0*
strides
*/
_output_shapes
:€€€€€€€€€   
П
3decoder_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_2/bias*
dtype0*
_output_shapes
: 
Ќ
$decoder_1/conv2d_transpose_2/BiasAddBiasAdd-decoder_1/conv2d_transpose_2/conv2d_transpose3decoder_1/conv2d_transpose_2/BiasAdd/ReadVariableOp*/
_output_shapes
:€€€€€€€€€   *
T0
Й
!decoder_1/conv2d_transpose_2/ReluRelu$decoder_1/conv2d_transpose_2/BiasAdd*/
_output_shapes
:€€€€€€€€€   *
T0
s
"decoder_1/conv2d_transpose_3/ShapeShape!decoder_1/conv2d_transpose_2/Relu*
T0*
_output_shapes
:
z
0decoder_1/conv2d_transpose_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2decoder_1/conv2d_transpose_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Њ
*decoder_1/conv2d_transpose_3/strided_sliceStridedSlice"decoder_1/conv2d_transpose_3/Shape0decoder_1/conv2d_transpose_3/strided_slice/stack2decoder_1/conv2d_transpose_3/strided_slice/stack_12decoder_1/conv2d_transpose_3/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
|
2decoder_1/conv2d_transpose_3/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
~
4decoder_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
∆
,decoder_1/conv2d_transpose_3/strided_slice_1StridedSlice"decoder_1/conv2d_transpose_3/Shape2decoder_1/conv2d_transpose_3/strided_slice_1/stack4decoder_1/conv2d_transpose_3/strided_slice_1/stack_14decoder_1/conv2d_transpose_3/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
|
2decoder_1/conv2d_transpose_3/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
~
4decoder_1/conv2d_transpose_3/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_3/strided_slice_2StridedSlice"decoder_1/conv2d_transpose_3/Shape2decoder_1/conv2d_transpose_3/strided_slice_2/stack4decoder_1/conv2d_transpose_3/strided_slice_2/stack_14decoder_1/conv2d_transpose_3/strided_slice_2/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
d
"decoder_1/conv2d_transpose_3/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ъ
 decoder_1/conv2d_transpose_3/mulMul,decoder_1/conv2d_transpose_3/strided_slice_1"decoder_1/conv2d_transpose_3/mul/y*
T0*
_output_shapes
: 
f
$decoder_1/conv2d_transpose_3/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ю
"decoder_1/conv2d_transpose_3/mul_1Mul,decoder_1/conv2d_transpose_3/strided_slice_2$decoder_1/conv2d_transpose_3/mul_1/y*
T0*
_output_shapes
: 
f
$decoder_1/conv2d_transpose_3/stack/3Const*
value	B :*
dtype0*
_output_shapes
: 
р
"decoder_1/conv2d_transpose_3/stackPack*decoder_1/conv2d_transpose_3/strided_slice decoder_1/conv2d_transpose_3/mul"decoder_1/conv2d_transpose_3/mul_1$decoder_1/conv2d_transpose_3/stack/3*
T0*
N*
_output_shapes
:
|
2decoder_1/conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
valueB: *
dtype0
~
4decoder_1/conv2d_transpose_3/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4decoder_1/conv2d_transpose_3/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
,decoder_1/conv2d_transpose_3/strided_slice_3StridedSlice"decoder_1/conv2d_transpose_3/stack2decoder_1/conv2d_transpose_3/strided_slice_3/stack4decoder_1/conv2d_transpose_3/strided_slice_3/stack_14decoder_1/conv2d_transpose_3/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
¶
<decoder_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp!decoder/conv2d_transpose_3/kernel*
dtype0*&
_output_shapes
: 
Ђ
-decoder_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput"decoder_1/conv2d_transpose_3/stack<decoder_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp!decoder_1/conv2d_transpose_2/Relu*
T0*
strides
*/
_output_shapes
:€€€€€€€€€@@*
paddingSAME
П
3decoder_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpdecoder/conv2d_transpose_3/bias*
dtype0*
_output_shapes
:
Ќ
$decoder_1/conv2d_transpose_3/BiasAddBiasAdd-decoder_1/conv2d_transpose_3/conv2d_transpose3decoder_1/conv2d_transpose_3/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@
r
decoder_1/Reshape_1/shapeConst*%
valueB"€€€€@   @      *
dtype0*
_output_shapes
:
Щ
decoder_1/Reshape_1Reshape$decoder_1/conv2d_transpose_3/BiasAdddecoder_1/Reshape_1/shape*
T0*/
_output_shapes
:€€€€€€€€€@@"&"• 
	variablesЧ Ф 
М
encoder/e1/kernel:0encoder/e1/kernel/Assign'encoder/e1/kernel/Read/ReadVariableOp:0(2.encoder/e1/kernel/Initializer/random_uniform:08
{
encoder/e1/bias:0encoder/e1/bias/Assign%encoder/e1/bias/Read/ReadVariableOp:0(2#encoder/e1/bias/Initializer/zeros:08
М
encoder/e2/kernel:0encoder/e2/kernel/Assign'encoder/e2/kernel/Read/ReadVariableOp:0(2.encoder/e2/kernel/Initializer/random_uniform:08
{
encoder/e2/bias:0encoder/e2/bias/Assign%encoder/e2/bias/Read/ReadVariableOp:0(2#encoder/e2/bias/Initializer/zeros:08
М
encoder/e3/kernel:0encoder/e3/kernel/Assign'encoder/e3/kernel/Read/ReadVariableOp:0(2.encoder/e3/kernel/Initializer/random_uniform:08
{
encoder/e3/bias:0encoder/e3/bias/Assign%encoder/e3/bias/Read/ReadVariableOp:0(2#encoder/e3/bias/Initializer/zeros:08
М
encoder/e4/kernel:0encoder/e4/kernel/Assign'encoder/e4/kernel/Read/ReadVariableOp:0(2.encoder/e4/kernel/Initializer/random_uniform:08
{
encoder/e4/bias:0encoder/e4/bias/Assign%encoder/e4/bias/Read/ReadVariableOp:0(2#encoder/e4/bias/Initializer/zeros:08
М
encoder/e5/kernel:0encoder/e5/kernel/Assign'encoder/e5/kernel/Read/ReadVariableOp:0(2.encoder/e5/kernel/Initializer/random_uniform:08
{
encoder/e5/bias:0encoder/e5/bias/Assign%encoder/e5/bias/Read/ReadVariableOp:0(2#encoder/e5/bias/Initializer/zeros:08
Ш
encoder/means/kernel:0encoder/means/kernel/Assign*encoder/means/kernel/Read/ReadVariableOp:0(21encoder/means/kernel/Initializer/random_uniform:08
З
encoder/means/bias:0encoder/means/bias/Assign(encoder/means/bias/Read/ReadVariableOp:0(2&encoder/means/bias/Initializer/zeros:08
†
encoder/log_var/kernel:0encoder/log_var/kernel/Assign,encoder/log_var/kernel/Read/ReadVariableOp:0(23encoder/log_var/kernel/Initializer/random_uniform:08
П
encoder/log_var/bias:0encoder/log_var/bias/Assign*encoder/log_var/bias/Read/ReadVariableOp:0(2(encoder/log_var/bias/Initializer/zeros:08
Ш
decoder/dense/kernel:0decoder/dense/kernel/Assign*decoder/dense/kernel/Read/ReadVariableOp:0(21decoder/dense/kernel/Initializer/random_uniform:08
З
decoder/dense/bias:0decoder/dense/bias/Assign(decoder/dense/bias/Read/ReadVariableOp:0(2&decoder/dense/bias/Initializer/zeros:08
†
decoder/dense_1/kernel:0decoder/dense_1/kernel/Assign,decoder/dense_1/kernel/Read/ReadVariableOp:0(23decoder/dense_1/kernel/Initializer/random_uniform:08
П
decoder/dense_1/bias:0decoder/dense_1/bias/Assign*decoder/dense_1/bias/Read/ReadVariableOp:0(2(decoder/dense_1/bias/Initializer/zeros:08
ƒ
!decoder/conv2d_transpose/kernel:0&decoder/conv2d_transpose/kernel/Assign5decoder/conv2d_transpose/kernel/Read/ReadVariableOp:0(2<decoder/conv2d_transpose/kernel/Initializer/random_uniform:08
≥
decoder/conv2d_transpose/bias:0$decoder/conv2d_transpose/bias/Assign3decoder/conv2d_transpose/bias/Read/ReadVariableOp:0(21decoder/conv2d_transpose/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_1/kernel:0(decoder/conv2d_transpose_1/kernel/Assign7decoder/conv2d_transpose_1/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_1/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_1/bias:0&decoder/conv2d_transpose_1/bias/Assign5decoder/conv2d_transpose_1/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_1/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_2/kernel:0(decoder/conv2d_transpose_2/kernel/Assign7decoder/conv2d_transpose_2/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_2/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_2/bias:0&decoder/conv2d_transpose_2/bias/Assign5decoder/conv2d_transpose_2/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_2/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_3/kernel:0(decoder/conv2d_transpose_3/kernel/Assign7decoder/conv2d_transpose_3/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_3/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_3/bias:0&decoder/conv2d_transpose_3/bias/Assign5decoder/conv2d_transpose_3/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_3/bias/Initializer/zeros:08"ѓ 
trainable_variablesЧ Ф 
М
encoder/e1/kernel:0encoder/e1/kernel/Assign'encoder/e1/kernel/Read/ReadVariableOp:0(2.encoder/e1/kernel/Initializer/random_uniform:08
{
encoder/e1/bias:0encoder/e1/bias/Assign%encoder/e1/bias/Read/ReadVariableOp:0(2#encoder/e1/bias/Initializer/zeros:08
М
encoder/e2/kernel:0encoder/e2/kernel/Assign'encoder/e2/kernel/Read/ReadVariableOp:0(2.encoder/e2/kernel/Initializer/random_uniform:08
{
encoder/e2/bias:0encoder/e2/bias/Assign%encoder/e2/bias/Read/ReadVariableOp:0(2#encoder/e2/bias/Initializer/zeros:08
М
encoder/e3/kernel:0encoder/e3/kernel/Assign'encoder/e3/kernel/Read/ReadVariableOp:0(2.encoder/e3/kernel/Initializer/random_uniform:08
{
encoder/e3/bias:0encoder/e3/bias/Assign%encoder/e3/bias/Read/ReadVariableOp:0(2#encoder/e3/bias/Initializer/zeros:08
М
encoder/e4/kernel:0encoder/e4/kernel/Assign'encoder/e4/kernel/Read/ReadVariableOp:0(2.encoder/e4/kernel/Initializer/random_uniform:08
{
encoder/e4/bias:0encoder/e4/bias/Assign%encoder/e4/bias/Read/ReadVariableOp:0(2#encoder/e4/bias/Initializer/zeros:08
М
encoder/e5/kernel:0encoder/e5/kernel/Assign'encoder/e5/kernel/Read/ReadVariableOp:0(2.encoder/e5/kernel/Initializer/random_uniform:08
{
encoder/e5/bias:0encoder/e5/bias/Assign%encoder/e5/bias/Read/ReadVariableOp:0(2#encoder/e5/bias/Initializer/zeros:08
Ш
encoder/means/kernel:0encoder/means/kernel/Assign*encoder/means/kernel/Read/ReadVariableOp:0(21encoder/means/kernel/Initializer/random_uniform:08
З
encoder/means/bias:0encoder/means/bias/Assign(encoder/means/bias/Read/ReadVariableOp:0(2&encoder/means/bias/Initializer/zeros:08
†
encoder/log_var/kernel:0encoder/log_var/kernel/Assign,encoder/log_var/kernel/Read/ReadVariableOp:0(23encoder/log_var/kernel/Initializer/random_uniform:08
П
encoder/log_var/bias:0encoder/log_var/bias/Assign*encoder/log_var/bias/Read/ReadVariableOp:0(2(encoder/log_var/bias/Initializer/zeros:08
Ш
decoder/dense/kernel:0decoder/dense/kernel/Assign*decoder/dense/kernel/Read/ReadVariableOp:0(21decoder/dense/kernel/Initializer/random_uniform:08
З
decoder/dense/bias:0decoder/dense/bias/Assign(decoder/dense/bias/Read/ReadVariableOp:0(2&decoder/dense/bias/Initializer/zeros:08
†
decoder/dense_1/kernel:0decoder/dense_1/kernel/Assign,decoder/dense_1/kernel/Read/ReadVariableOp:0(23decoder/dense_1/kernel/Initializer/random_uniform:08
П
decoder/dense_1/bias:0decoder/dense_1/bias/Assign*decoder/dense_1/bias/Read/ReadVariableOp:0(2(decoder/dense_1/bias/Initializer/zeros:08
ƒ
!decoder/conv2d_transpose/kernel:0&decoder/conv2d_transpose/kernel/Assign5decoder/conv2d_transpose/kernel/Read/ReadVariableOp:0(2<decoder/conv2d_transpose/kernel/Initializer/random_uniform:08
≥
decoder/conv2d_transpose/bias:0$decoder/conv2d_transpose/bias/Assign3decoder/conv2d_transpose/bias/Read/ReadVariableOp:0(21decoder/conv2d_transpose/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_1/kernel:0(decoder/conv2d_transpose_1/kernel/Assign7decoder/conv2d_transpose_1/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_1/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_1/bias:0&decoder/conv2d_transpose_1/bias/Assign5decoder/conv2d_transpose_1/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_1/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_2/kernel:0(decoder/conv2d_transpose_2/kernel/Assign7decoder/conv2d_transpose_2/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_2/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_2/bias:0&decoder/conv2d_transpose_2/bias/Assign5decoder/conv2d_transpose_2/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_2/bias/Initializer/zeros:08
ћ
#decoder/conv2d_transpose_3/kernel:0(decoder/conv2d_transpose_3/kernel/Assign7decoder/conv2d_transpose_3/kernel/Read/ReadVariableOp:0(2>decoder/conv2d_transpose_3/kernel/Initializer/random_uniform:08
ї
!decoder/conv2d_transpose_3/bias:0&decoder/conv2d_transpose_3/bias/Assign5decoder/conv2d_transpose_3/bias/Read/ReadVariableOp:0(23decoder/conv2d_transpose_3/bias/Initializer/zeros:08*Й
reconstructionsv
6
images,
Placeholder:0€€€€€€€€€@@<
images2
decoder/Reshape_1:0€€€€€€€€€@@*Е
decoderz
8
latent_vectors&
Placeholder_1:0€€€€€€€€€
>
images4
decoder_1/Reshape_1:0€€€€€€€€€@@*Ѕ
gaussian_encoderђ
6
images,
Placeholder:0€€€€€€€€€@@6
mean.
encoder/means/BiasAdd:0€€€€€€€€€
:
logvar0
encoder/log_var/BiasAdd:0€€€€€€€€€
