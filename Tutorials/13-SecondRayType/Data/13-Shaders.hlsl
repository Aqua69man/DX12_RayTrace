/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***************************************************************************/
RaytracingAccelerationStructure gRtScene : register(t0);
RWTexture2D<float4> gOutput : register(u0);

cbuffer PerFrame : register(b0)
{
    float3 A;
    float3 B;
    float3 C;
}

float3 linearToSrgb(float3 c)
{
    // Based on http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
    float3 sq1 = sqrt(c);
    float3 sq2 = sqrt(sq1);
    float3 sq3 = sqrt(sq2);
    float3 srgb = 0.662002687 * sq1 + 0.684122060 * sq2 - 0.323583601 * sq3 - 0.0225411470 * c;
    return srgb;
}

struct RayPayload
{
    float3 color;
};

[shader("raygeneration")]
void rayGen()
{
    uint3 launchIndex = DispatchRaysIndex();
    uint3 launchDim = DispatchRaysDimensions();

    float2 crd = float2(launchIndex.xy);
    float2 dims = float2(launchDim.xy);

    float2 d = ((crd/dims) * 2.f - 1.f);
    float aspectRatio = dims.x / dims.y;

    RayDesc ray;
    ray.Origin = float3(0, 0, -2);
    ray.Direction = normalize(float3(d.x * aspectRatio, -d.y, 1));

    ray.TMin = 0;
    ray.TMax = 100000;

    RayPayload payload;
	// TraceRay():
	//     param_1  -  Is the TLAS SRV.
	//     param_2  -  Is the ray flags. These flags allow us to control the traversal behavior,
	//	    		       for example enable back-face culling.
	//     param_3	-  Is the ray-mask. It can be used to cull entire objects when tracing rays.
	//	    		       We will not cover this topic in the tutorials. 0xFF means no culling.
	//     param_4  -  RayContributionToHitGroupIndex - the ray-index. 0 For the primary-ray, 1 for the shadow-ray. 
	//     param_5  -  MultiplierForGeometryContributionToShaderIndex - only affects instances with multiple geometries in BLAS. 
	//				       In our case, it affects only BLAS with two geometries (Triangle + Plane). 
	//					   GeometryIndex(for Triangle) = 0, GeometryIndex(for Plane) = 1 
	//					   Actually, this is the distance in records between geometries. 
	//     param_6	 - Is the MissShaderIndex. This index is relative to the base miss-shader index we passed when 
	//	    			   calling DispatchRays(). Since our miss-shaders entries are stored contiguously 
	//					   in the shader-table, we can treat this value as the ray-index.
	//     param_7	 - Is the RayDesc object we created.
	//     param_8	 - RayPayload object.
	
	// MISS_SHADER_ID = missOffset(from_DispatchRays=1) + MissShaderIndex(param_6)
	//	   - missOffset: from_DispatchRays = 1
	//     - MissShaderIndex: leaving = 0
	// => MISS_SHADER_ID = 1 + 0 = 1

	// PRIMARY_HIT_SHADER_ID = hitOffset(from_DispatchRays=3)
	//						   InstanceContributionToHitGroupIndex(from_TLAS) + 
	//						   GeometryIndex(from_mpVertexBuffer=0/1) * MultiplierForGeometryContributionToShaderIndex(param_5) +
	//						   RayContributionToHitGroupIndex(param_4)
	//
	// RayContributionToHitGroupIndex:						Leaving  = 0 (This is the first/PRIMARY ray)
	// MultiplierForGeometryContributionToShaderIndex:		SETTING  = 2

	// We have 11 entries(SHADER_ID):
	//     Entry 0 - Ray - gen program
	// 	   Entry 1 - Miss program for the primary ray
	// 	   Entry 2 - Miss program for the shadow ray
	// 	   Entries 3, 4 - Hit programs for triangle 0 (primary followed by shadow)
	// 	   Entries 5, 6 - Hit programs for the plane(primary followed by shadow)
	// 	   Entries 7, 8 - Hit programs for triangle 1 (primary followed by shadow)
	// 	   Entries 9, 10 - Hit programs for triangle 2 (primary followed by shadow)

    TraceRay( gRtScene, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, 2 /*!-HERE-!*/, 0, ray, payload );
    float3 col = linearToSrgb(payload.color);
    gOutput[launchIndex.xy] = float4(col, 1);
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    payload.color = float3(0.4, 0.6, 0.2);
}

[shader("closesthit")]
void triangleChs(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    payload.color = A * barycentrics.x + B * barycentrics.y + C * barycentrics.z;
}

struct ShadowPayload
{
    bool hit;
};

[shader("closesthit")]
void planeChs(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    float hitT = RayTCurrent();	// ray distance, i.e. distance from ray's origin to intersection point
    float3 rayDirW = WorldRayDirection(); // WorldSpace direction of incoming ray: (hitPoint - originPoint)
    float3 rayOriginW = WorldRayOrigin(); // WorldSpace originPoint position of incoming ray

    // Find the world-space hit position
    float3 posW = rayOriginW + hitT * rayDirW;

    // Fire a shadow ray. The direction is hard-coded here, but can be fetched from a constant-buffer
    RayDesc ray;
    ray.Origin = posW;
    ray.Direction = normalize(float3(0.5, 0.5, -0.5));  // hard-coded direction to the light source
    ray.TMin = 0.01; // !!_Note_!! that we do not use 0 for TMin but set it into a small value. This is to avoid aliasing issues due to floating-point errors.
    ray.TMax = 100000;
    ShadowPayload shadowPayload;

	// TraceRay():
	//     param_1  -  Is the TLAS SRV.
	//     param_2  -  Is the ray flags. These flags allow us to control the traversal behavior,
	//	    		       for example enable back-face culling.
	//     param_3	-  Is the ray-mask. It can be used to cull entire objects when tracing rays.
	//	    		       We will not cover this topic in the tutorials. 0xFF means no culling.
	//     param_4  -  RayContributionToHitGroupIndex - the ray-index. 0 For the primary-ray, 1 for the shadow-ray. 
	//     param_5  -  MultiplierForGeometryContributionToShaderIndex - only affects instances with multiple geometries in BLAS. 
	//				       In our case, it affects only BLAS with two geometries (Triangle + Plane). 
	//					   GeometryIndex(for Triangle) = 0, GeometryIndex(for Plane) = 1 
	//					   Actually, this is the distance in records between geometries. 
	//     param_6	 - Is the MissShaderIndex. This index is relative to the base miss-shader index we passed when 
	//	    			   calling DispatchRays(). Since our miss-shaders entries are stored contiguously 
	//					   in the shader-table, we can treat this value as the ray-index.
	//     param_7	 - Is the RayDesc object we created.
	//     param_8	 - RayPayload object.

	// MISS_SHADER_ID = missOffset(from_DispatchRays) + MissShaderIndex (param_6)
	//	   - missOffset(from_DispatchRays) = 1
	//     - MissShaderIndex: SETTING = 1
	// => MISS_SHADER_ID = 1 + 1 = 2

	// PRIMARY_HIT_SHADER_ID = hitOffset(from_DispatchRays=3)
	//						   InstanceContributionToHitGroupIndex(from_TLAS) + 
	//						   GeometryIndex(from_mpVertexBuffer=0/1) * MultiplierForGeometryContributionToShaderIndex(param_5) +
	//						   RayContributionToHitGroupIndex(param_4)
	//
	// RayContributionToHitGroupIndex:						SETTING  = 1 (This is the second/SHADOW ray - previos ray had index=0, for this one has index=1)
	// MultiplierForGeometryContributionToShaderIndex:		Leaving  = 0

	// We have 11 entries(SHADER_ID):
	//     Entry 0 - Ray - gen program
	// 	   Entry 1 - Miss program for the primary ray
	// 	   Entry 2 - Miss program for the shadow ray
	// 	   Entries 3, 4 - Hit programs for triangle 0 (primary followed by shadow)
	// 	   Entries 5, 6 - Hit programs for the plane(primary followed by shadow)
	// 	   Entries 7, 8 - Hit programs for triangle 1 (primary followed by shadow)
	// 	   Entries 9, 10 - Hit programs for triangle 2 (primary followed by shadow)

    TraceRay(gRtScene, 0  /*rayFlags*/, 0xFF, 1 /* ray index*/, 0, 1 /*!-HERE-!*/, ray, shadowPayload);

    float factor = shadowPayload.hit ? 0.1 : 1.0;
    payload.color = float4(0.9f, 0.9f, 0.9f, 1.0f) * factor;
}

[shader("closesthit")]
void shadowChs(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.hit = true;
}

[shader("miss")]
void shadowMiss(inout ShadowPayload payload)
{
    payload.hit = false;
}
