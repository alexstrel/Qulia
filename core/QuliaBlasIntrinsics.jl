module QuliaBlasIntrinsics

using Base:llvmcall

# https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=FMA

# Plain baseline registers
const half2   = NTuple{2, VecElement{Float16}}
const half4   = NTuple{4, VecElement{Float16}}
#
const float2  = NTuple{2, VecElement{Float32}}
const float4  = NTuple{4, VecElement{Float32}}
const float8  = NTuple{8, VecElement{Float32}}
const float16 = NTuple{16,VecElement{Float32}}
#
const double2 = NTuple{2, VecElement{Float64}}
const double4 = NTuple{4, VecElement{Float64}}
const double8 = NTuple{8, VecElement{Float64}}

@inline prefetch(a, i1, i2, i3) = ccall("llvm.prefetch", llvmcall, Cvoid, (Ptr{Cchar}, Int32, Int32, Int32), a, i1, i2, i3)

@inline prefetchT0(a) = mmprefetch(a, Int32(0), Int32(3), Int(1))
@inline prefetchT1(a) = mmprefetch(a, Int32(0), Int32(2), Int(1))
@inline prefetchT2(a) = mmprefetch(a, Int32(0), Int32(1), Int(1))

@inline function neg(x::double4)
    llvmcall("""
        %res = fneg <4 x double> %0
        ret <4 x double> %res
        """, double4, Tuple{double4,}, x)
end

@inline function neg(x::double8)
    llvmcall("""
        %res = fneg <8 x double> %0
        ret <8 x double> %res
        """, double8, Tuple{double8,}, x)
end

@inline function neg(x::float8)
    llvmcall("""
        %res = fneg <8 x float> %0
        ret <8 x float> %res
        """, float8, Tuple{float8,}, x)
end

@inline function neg(x::float16)
    llvmcall("""
        %res = fneg <16 x float> %0
        ret <16 x float> %res
        """, float16, Tuple{float16,}, x)
end

@inline function fma(x::double4, y::double4, z::double4)
  out = ccall("llvm.fma.v4f64", llvmcall, double4, (double4, double4, double4), x, y, z)
end

@inline function fma(x::double8, y::double8, z::double8) 
  out = ccall("llvm.fma.v8f64", llvmcall, double8, (double8, double8, double8), x, y, z)
end

@inline function fma(x::float8, y::float8, z::float8) 
  out = ccall("llvm.fma.v8f32", llvmcall, float8, (float8, float8, float8), x, y, z)
end

@inline function fma(x::float16, y::float16, z::float16)
  out = ccall("llvm.fma.v16f32", llvmcall, float16, (float16, float16, float16), x, y, z)
end


end #QuliaBlasIntrinsics
