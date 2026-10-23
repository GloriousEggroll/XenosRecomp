#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

#if !defined(__cplusplus) || defined(__INTELLISENSE__)

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

#ifdef __spirv__

struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;

#define g_Booleans                 vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 260)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 264)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 272)

#ifdef GOW3_RECOMP
// Gears ABI: 32 descriptors per texture dimension/sampler, eight boolean
// words, presentation constants, then 32 packed loop words at offset 576.
// Host layout must match.
#undef g_Booleans
#undef g_SwappedTexcoords
#undef g_HalfPixelOffset
#undef g_AlphaThreshold
#define g_SwappedTexcoords vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 544)
#define g_HalfPixelOffset vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 548)
#define g_AlphaThreshold vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 556)
#define g_ColorScale vk::RawBufferLoad<float4>(g_PushConstants.SharedConstants + 560)
#endif

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;

#define g_SpecConstants() g_SpecConstants

#else

#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x);

uint g_SpecConstants();

#endif

Texture2D<float4> g_Texture2DDescriptorHeap[] : register(t0, space0);
Texture3D<float4> g_Texture3DDescriptorHeap[] : register(t0, space1);
TextureCube<float4> g_TextureCubeDescriptorHeap[] : register(t0, space2);
SamplerState g_SamplerDescriptorHeap[] : register(s0, space3);

uint2 getTexture2DDimensions(Texture2D<float4> texture)
{
    uint2 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y);
    return dimensions;
}

float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture));
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DDimensions(texture) + offset - 0.5));
}

float w0(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a)
{
    return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a)
{
    return (1.0f / 6.0f) * (a * a * a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

float4 tfetch2DBicubic(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    uint2 dimensions = getTexture2DDimensions(texture);
    
    float x = texCoord.x * dimensions.x + offset.x;
    float y = texCoord.y * dimensions.y + offset.y;

    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float4 r =
        g0(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h0y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h0y) / float2(dimensions))) +
        g1(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h1y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h1y) / float2(dimensions)));

    return r;
}

float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord);
}

struct CubeMapData
{
    float3 cubeMapDirections[2];
    uint cubeMapIndex;
};

#ifdef GOW3_RECOMP
// Matches src/renderer/native/shader_abi.h. One descriptor per Xenos vertex
// fetch constant: native address, byte size, endian mode. Full/mini fetches
// share their indexed word offset in the shader, as the hardware does.
uint gow3VertexWord(uint slot, int word)
{
    uint64_t descriptor = g_PushConstants.SharedConstants + 704 + slot * 16;
    uint64_t address = vk::RawBufferLoad<uint64_t>(descriptor, 8);
    uint size = vk::RawBufferLoad<uint>(descriptor + 8);
    uint endian = vk::RawBufferLoad<uint>(descriptor + 12);
    if (address == 0 || word < 0 || uint(word) >= size / 4) return 0;
    uint value = vk::RawBufferLoad<uint>(address + uint(word) * 4);
    if (endian == 1 || endian == 2)
        value = ((value & 0x00ff00ff) << 8) | ((value >> 8) & 0x00ff00ff);
    if (endian == 2 || endian == 3) value = (value << 16) | (value >> 16);
    return value;
}

float gow3VertexInteger(uint word, uint shift, uint bits, bool isSigned, bool integer, bool noZero)
{
    uint raw = bits == 32 ? word : ((word >> shift) & ((1u << bits) - 1));
    float value = isSigned ? float(int(raw << (32 - bits)) >> (32 - bits)) : float(raw);
    if (integer) return value;
    float maximum = isSigned ? exp2(float(bits - 1)) - 1.0 : exp2(float(bits)) - 1.0;
    if (isSigned && noZero) return (value + 0.5) / (maximum + 0.5);
    return isSigned ? max(value / maximum, -1.0) : value / maximum;
}

float gow3VertexHalf(uint value)
{
    // Xenos half precision has a finite exponent-31 range (no Inf/NaN).
    if ((value & 0x7c00) == 0x7c00)
        return ((value & 0x8000) ? -1.0 : 1.0) * (65536.0 + (value & 1023) * 64.0);
    return f16tof32(value & 65535);
}

float4 gow3FetchVertex(uint slot, int word, uint format, bool isSigned, bool integer, bool noZero, int exponent)
{
    float4 result = 0;
    uint4 widths = 0, shifts = 0, indices = 0;
    switch (format) {
    case 6: widths = 8; shifts = uint4(0, 8, 16, 24); break;
    case 7: widths = uint4(10, 10, 10, 2); shifts = uint4(0, 10, 20, 30); break;
    case 16: widths = uint4(11, 11, 10, 0); shifts = uint4(0, 11, 22, 0); break;
    case 17: widths = uint4(10, 11, 11, 0); shifts = uint4(0, 10, 21, 0); break;
    case 25: widths = uint4(16, 16, 0, 0); shifts.y = 16; break;
    case 26: widths = 16; shifts = uint4(0, 16, 0, 16); indices.zw = 1; break;
    case 31: case 32: {
        uint a = gow3VertexWord(slot, word);
        result.xy = float2(gow3VertexHalf(a), gow3VertexHalf(a >> 16));
        if (format == 32) {
            uint b = gow3VertexWord(slot, word + 1);
            result.zw = float2(gow3VertexHalf(b), gow3VertexHalf(b >> 16));
        }
        return result * exp2(float(exponent));
    }
    case 33: widths.x = 32; break;
    case 34: widths.xy = 32; indices.y = 1; break;
    case 35: widths = 32; indices = uint4(0, 1, 2, 3); break;
    case 36: case 37: case 38: case 57: {
        uint count = format == 57 ? 3 : (format == 38 ? 4 : format - 35);
        [unroll] for (uint i = 0; i < 4; ++i)
            if (i < count) result[i] = asfloat(gow3VertexWord(slot, word + i));
        return result * exp2(float(exponent));
    }
    }
    [unroll] for (uint c = 0; c < 4; ++c)
        if (widths[c] != 0) result[c] = gow3VertexInteger(gow3VertexWord(slot, word + indices[c]),
            shifts[c], widths[c], isSigned, integer, noZero);
    return result * exp2(float(exponent));
}

// Texture instructions select implicit derivatives, explicit gradients or an
// explicit mip level. Keep that choice in the shader, independent of the host
// sampler's min/mag/mip filters. Resource uploads represent 1D as 2D height 1.
float gow3GammaToLinear(float gamma)
{
    // Xenos piecewise gamma, matching the BSD Xenia/ReXGlue implementation
    // (not host sRGB). See src/renderer/native/texture_data.LICENSE.
    gamma = isnan(gamma) ? 0.0 : saturate(gamma);
    float scale = gamma >= 192.0 / 255.0 ? 8.0 / 1024.0 :
                  gamma >= 96.0 / 255.0 ? 4.0 / 1024.0 :
                  gamma >= 64.0 / 255.0 ? 2.0 / 1024.0 : 1.0 / 1024.0;
    float offset = gamma >= 192.0 / 255.0 ? -1024.0 :
                   gamma >= 96.0 / 255.0 ? -256.0 :
                   gamma >= 64.0 / 255.0 ? -64.0 : 0.0;
    precise float converted = gamma * (255.0 * 1024.0) * scale + offset;
    converted += trunc(converted * scale);
    return converted / 1023.0;
}

float4 gow3TextureResult(float4 value, uint slot)
{
    uint2 properties = vk::RawBufferLoad<uint2>(g_PushConstants.SharedConstants + 2240 + slot * 8);
    [unroll] for (uint channel = 0; channel < 4; ++channel) {
        uint sign = (properties.x >> (channel * 2)) & 3;
        if (sign == 2) value[channel] = value[channel] * 2.0 - 1.0;
        if (sign == 3) value[channel] = gow3GammaToLinear(value[channel]);
    }
    return value * exp2(float(asint(properties.y)));
}

float4 gow3Fetch2D(uint resource, uint samplerIndex, float2 coord, float2 offset,
    bool unnormalized, bool computed, bool gradients, float lod,
    float3 gradH, float3 gradV)
{
    Texture2D<float4> tex = g_Texture2DDescriptorHeap[resource];
    SamplerState state = g_SamplerDescriptorHeap[samplerIndex];
    float2 size = getTexture2DDimensions(tex);
    coord = (unnormalized ? coord / size : coord) + offset / size;
    if (computed) {
        if (gradients) {
            float2 scale = exp2(lod) / (unnormalized ? size : 1.0);
            return tex.SampleGrad(state, coord, gradH.xy * scale, gradV.xy * scale);
        }
#ifndef GOW3_VERTEX_SHADER
        return tex.SampleBias(state, coord, lod);
#endif
    }
    return tex.SampleLevel(state, coord, lod);
}

float4 gow3Fetch3D(uint resource, uint samplerIndex, float3 coord, float3 offset,
    bool unnormalized, bool computed, bool gradients, float lod,
    float3 gradH, float3 gradV)
{
    Texture3D<float4> tex = g_Texture3DDescriptorHeap[resource];
    SamplerState state = g_SamplerDescriptorHeap[samplerIndex];
    uint width, height, depth;
    tex.GetDimensions(width, height, depth);
    float3 size = float3(width, height, depth);
    coord = (unnormalized ? coord / size : coord) + offset / size;
    if (computed) {
        if (gradients) {
            float3 scale = exp2(lod) / (unnormalized ? size : 1.0);
            return tex.SampleGrad(state, coord, gradH * scale, gradV * scale);
        }
#ifndef GOW3_VERTEX_SHADER
        return tex.SampleBias(state, coord, lod);
#endif
    }
    return tex.SampleLevel(state, coord, lod);
}

float4 gow3FetchCube(uint resource, uint samplerIndex, float3 coord,
    bool computed, bool gradients, float lod, float3 gradH, float3 gradV)
{
    TextureCube<float4> tex = g_TextureCubeDescriptorHeap[resource];
    SamplerState state = g_SamplerDescriptorHeap[samplerIndex];
    if (computed) {
        if (gradients) return tex.SampleGrad(state, coord, gradH * exp2(lod), gradV * exp2(lod));
#ifndef GOW3_VERTEX_SHADER
        return tex.SampleBias(state, coord, lod);
#endif
    }
    return tex.SampleLevel(state, coord, lod);
}

float4 gow3Weights2D(uint resource, uint samplerIndex, float2 coord, float2 offset, bool unnormalized)
{
    float2 size = getTexture2DDimensions(g_Texture2DDescriptorHeap[resource]);
    float2 weights = frac((unnormalized ? coord : coord * size) + offset - 0.5);
    return float4(select(isnan(coord), 0.0, weights), 0, 0);
}
#endif

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]);
}

float4 tfetchR11G11B10(uint4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        return float4(
            (value.x & 0x00000400 ? -1.0 : 0.0) + ((value.x & 0x3FF) / 1024.0),
            (value.x & 0x00200000 ? -1.0 : 0.0) + (((value.x >> 11) & 0x3FF) / 1024.0),
            (value.x & 0x80000000 ? -1.0 : 0.0) + (((value.x >> 22) & 0x1FF) / 512.0),
            0.0);
    }
    else
    {
        return asfloat(value);
    }
}

float4 tfetchTexcoord(uint swappedTexcoords, float4 value, uint semanticIndex)
{
    return (swappedTexcoords & (1ull << semanticIndex)) != 0 ? value.yxwz : value;
}

float4 cube(float4 value, inout CubeMapData cubeMapData)
{
    uint index = cubeMapData.cubeMapIndex;
    cubeMapData.cubeMapDirections[index] = value.xyz;
    ++cubeMapData.cubeMapIndex;
    
    return float4(0.0, 0.0, 0.0, index);
}

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

float4 max4(float4 src0)
{
    return max(max(src0.x, src0.y), max(src0.z, src0.w));
}

float2 getPixelCoord(uint resourceDescriptorIndex, float2 texCoord)
{
    return getTexture2DDimensions(g_Texture2DDescriptorHeap[resourceDescriptorIndex]) * texCoord;
}

float computeMipLevel(float2 pixelCoord)
{
    float2 dx = ddx(pixelCoord);
    float2 dy = ddy(pixelCoord);
    float deltaMaxSqr = max(dot(dx, dx), dot(dy, dy));
    return max(0.0, 0.5 * log2(deltaMaxSqr));
}

#endif

#endif
