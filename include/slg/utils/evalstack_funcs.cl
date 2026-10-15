#line 2 "evalstack_funcs.cl"

/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

// Macros used mostly for Texture and Material evluations

// The material and texture evaluation stacks are in global memory, a block of
// maxMaterialEvalStackSize (or maxTextureEvalStackSize) floats per task, or,
// on HIP when the scene needs small enough stacks, in private arrays of
// PARAM_MATERIAL_EVALSTACK_SIZE (or PARAM_TEXTURE_EVALSTACK_SIZE) floats (see
// PathOCLBaseOCLRenderThread::InitKernels()). The evaluation functions take
// the stack as a __global pointer, which is the generic address space on HIP.
#if defined(LUXRAYS_CUDA_DEVICE) && defined(PARAM_EVALSTACK_PRIVATE)
#define EvalStack_Declare(stacks, stackSize, privateStackSize) \
	float evalStackPrivate[privateStackSize]; \
	__global float *evalStack = evalStackPrivate
#else
#define EvalStack_Declare(stacks, stackSize, privateStackSize) \
	__global float *evalStack = &stacks[get_global_id(0) * stackSize]
#endif

#define EvalStack_PushInt(a) { ((__global int *)evalStack)[*evalStackOffset] = a; *evalStackOffset = *evalStackOffset + 1; }
#define EvalStack_PopInt(a) { *evalStackOffset = *evalStackOffset - 1; a = ((__global int *)evalStack)[*evalStackOffset]; }

#define EvalStack_PushUInt(a) { ((__global uint *)evalStack)[*evalStackOffset] = a; *evalStackOffset = *evalStackOffset + 1; }
#define EvalStack_PopUInt(a) { *evalStackOffset = *evalStackOffset - 1; a = ((__global uint *)evalStack)[*evalStackOffset]; }

#define EvalStack_PushFloat(a) { evalStack[*evalStackOffset] = a; *evalStackOffset = *evalStackOffset + 1; }
#define EvalStack_PushFloat2(a) { EvalStack_PushFloat(a.x); EvalStack_PushFloat(a.y); }
#define EvalStack_PushFloat3(a) { EvalStack_PushFloat(a.x); EvalStack_PushFloat(a.y); EvalStack_PushFloat(a.z); }
#define EvalStack_PopFloat(a) { *evalStackOffset = *evalStackOffset - 1; a = evalStack[*evalStackOffset]; }
#define EvalStack_PopFloat2(a) { EvalStack_PopFloat(a.y); EvalStack_PopFloat(a.x); }
#define EvalStack_PopFloat3(a) { EvalStack_PopFloat(a.z); EvalStack_PopFloat(a.y); EvalStack_PopFloat(a.x); }
#define EvalStack_ReadFloat(x) (evalStack[(*evalStackOffset) + x])
#define EvalStack_ReadFloat2(x) (MAKE_FLOAT2(evalStack[(*evalStackOffset) + x], evalStack[(*evalStackOffset) + x + 1]))
#define EvalStack_ReadFloat3(x) (MAKE_FLOAT3(evalStack[(*evalStackOffset) + x], evalStack[(*evalStackOffset) + x + 1], evalStack[(*evalStackOffset) + x + 2]))

#define EvalStack_PushBSDFEvent(a) { ((__global BSDFEvent *)evalStack)[*evalStackOffset] = a; *evalStackOffset = *evalStackOffset + 1; }
#define EvalStack_PopBSDFEvent(a) { *evalStackOffset = *evalStackOffset - 1; a = ((__global BSDFEvent *)evalStack)[*evalStackOffset]; }
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
