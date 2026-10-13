/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
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

#if !defined(LUXRAYS_DISABLE_HIP)

#ifndef _LUXRAYS_HIPRTACCEL_H
#define	_LUXRAYS_HIPRTACCEL_H

#include "luxrays/luxrays.h"
#include "luxrays/core/accelerator.h"

namespace luxrays {

class HiprtAccel : public Accelerator {
public:
	HiprtAccel(const Context & context);
	virtual ~HiprtAccel();

	virtual AcceleratorType GetType() const { return ACCEL_HIPRT; }

	virtual bool HasNativeSupport(const IntersectionDevice &device) const;
	virtual bool HasHWSupport(const IntersectionDevice &device) const;

	virtual HardwareIntersectionKernelUPtr NewHardwareIntersectionKernel(HardwareIntersectionDevice &device) const override;

	virtual void Init(const std::deque<const Mesh * > &meshes,
		const u_longlong totalVertexCount,
		const u_longlong totalTriangleCount);

	virtual bool Intersect(const Ray *ray, RayHit *hit) const;

	friend class HiprtKernel;

private:
	const Context & ctx;
	std::deque<const Mesh * > meshes;
	u_longlong totalVertexCount, totalTriangleCount;

	bool initialized;
};

}

#endif

#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
