module QuliaComplex

import QuliaBlasIntrinsics

mutable struct PackedComplex{N, T <: AbstractFloat}
  real::NTuple{N,  VecElement{T}}
  imag::NTuple{N,  VecElement{T}}

  PackedComplex{N, T}(r::NTuple{N,  VecElement{T}}, i::NTuple{N,  VecElement{T}}) where {N} where {T <: AbstractFloat} = new(r, i)
  PackedComplex{N, T}(r::NTuple{N,  VecElement{T}}) where {N} where {T <: AbstractFloat} = new(r, NTuple{N,  VecElement{T}}(i->0, N))
end 

const pcdouble4 = PackedComplex{4, Float64}
const pcdouble8 = PackedComplex{8, Float64}
#
const pcfloat4  = PackedComplex{8 , Float32}
const pcfloat8  = PackedComplex{16, Float32}

struct ComplexFMA{N, T <: AbstractFloat}
end

@inline function ComplexFMA{N, T}(x::PackedComplex{N, T}, y::PackedComplex{N, T}, z::PackedComplex{N, T}) where {N} where{T <: AbstractFloat}
  #
  r      = QuliaBlasIntrinsics.fma(x.real, y.real, z.real)
  negimx = QuliaBlasIntrinsics.neg(x.imag)
  r      = QuliaBlasIntrinsics.fma(negimx, y.imag, r     )
  i      = QuliaBlasIntrinsics.fma(x.real, y.imag, z.imag)
  i      = QuliaBlasIntrinsics.fma(x.imag, y.real, i     )  
  #
  return PackedComplex{N, T}(r, i)
end

cfma_d4 =ComplexFMA{4 , Float64}
cfma_d8 =ComplexFMA{8 , Float64}

cfma_f8 =ComplexFMA{8 , Float64}
cfma_f16=ComplexFMA{16, Float64}

@inline function fma(x::pcdouble4, y::pcdouble4, z::pcdouble4)
   cfma_d4(x, y, z)
end

@inline function fma(x::pcdouble8, y::pcdouble8, z::pcdouble8)
   cfma_d8(x, y, z)
end

@inline function fma(x::pcfloat8, y::pcfloat8, z::pcfloat8)
   cfma_f8(x, y, z)
end

@inline function fma(x::pcfloat16, y::pcfloat16, z::pcfloat16)
   cfma_f16(x, y, z)
end


end #QuliaComplex
