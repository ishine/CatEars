// gemm_penryn.cc -- Created at 2018-08-12
// Use the code of blis
/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "gemm.h"
#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

namespace pocketkaldi {

void bli_sgemm_penryn_asm_8x4
     (
       uint64_t               k0,
       float*      alpha,
       float*      a,
       float*      b,
       float*      beta,
       float*      c, uint64_t rs_c0, uint64_t cs_c0,
       const float* b_next
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	// void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()
		
		
		mov(var(a), rax) // load address of a.
		mov(var(b), rbx) // load address of b.
		mov(var(b_next), r9) // load address of b_next.
		
		sub(imm(0-8*16), rax) // increment pointers to allow byte
		sub(imm(0-8*16), rbx) // offsets in the unrolled iterations.
		
		movaps(mem(rax, -8*16), xmm0) // initialize loop by pre-loading elements
		movaps(mem(rax, -7*16), xmm1) // of a and b.
		movaps(mem(rbx, -8*16), xmm2)
		
		mov(var(c), rcx) // load address of c
		mov(var(cs_c), rdi) // load cs_c
		lea(mem(, rdi, 4), rdi) // cs_c *= sizeof(float)
		mov(rdi, r12) // make a copy of cs_c (in bytes)
		lea(mem(rcx, rdi, 2), r10) // load address of c + 2*cs_c;
		
		prefetch(2, mem(r9, 0*4)) // prefetch b_next
		
		xorps(xmm3, xmm3)
		xorps(xmm4, xmm4)
		xorps(xmm5, xmm5)
		xorps(xmm6, xmm6)
		
		prefetch(2, mem(rcx, 6*4)) // prefetch c + 0*cs_c
		xorps(xmm8, xmm8)
		xorps(xmm9, xmm9)
		prefetch(2, mem(rcx, rdi, 1, 6*4)) // prefetch c + 1*cs_c
		xorps(xmm10, xmm10)
		xorps(xmm11, xmm11)
		prefetch(2, mem(r10, 6*4)) // prefetch c + 2*cs_c
		xorps(xmm12, xmm12)
		xorps(xmm13, xmm13)
		prefetch(2, mem(r10, rdi, 1, 6*4)) // prefetch c + 3*cs_c
		xorps(xmm14, xmm14)
		xorps(xmm15, xmm15)
		
		
		
		mov(var(k_iter), rsi) // i = k_iter;
		test(rsi, rsi) // check i via logical AND.
		je(.SCONSIDKLEFT) // if i == 0, jump to code that
		 // contains the k_left loop.
		
		
		label(.SLOOPKITER) // MAIN LOOP
		
		prefetch(0, mem(rax, (4*35+1)*8))
		
		addps(xmm6, xmm10) // iteration 0
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)
		
		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)
		
		addps(xmm2, xmm8)
		movaps(mem(rbx, -7*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)
		
		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -6*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -5*16), xmm1)
		
		
		addps(xmm6, xmm10) // iteration 1
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)
		
		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)
		
		addps(xmm2, xmm8)
		movaps(mem(rbx, -6*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)
		
		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -4*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -3*16), xmm1)
		
		
		addps(xmm6, xmm10) // iteration 2
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)
		
		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)
		
		addps(xmm2, xmm8)
		movaps(mem(rbx, -5*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)
		
		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -2*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -1*16), xmm1)
		
		
		addps(xmm6, xmm10) // iteration 3
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)
		
		sub(imm(0-4*8*4), rax) // a += 4*8 (unroll x mr)
		
		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)
		
		sub(imm(0-4*4*4), r9) // b_next += 4*4 (unroll x nr)
		
		addps(xmm2, xmm8)
		movaps(mem(rbx, -4*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)
		
		sub(imm(0-4*4*4), rbx) // b += 4*4 (unroll x nr)
		
		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -8*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -7*16), xmm1)
		
		prefetch(2, mem(r9, 0*4)) // prefetch b_next[0]
		prefetch(2, mem(r9, 16*4)) // prefetch b_next[16]
		
		
		dec(rsi) // i -= 1;
		jne(.SLOOPKITER) // iterate again if i != 0.
		
		
		
		label(.SCONSIDKLEFT)
		
		mov(var(k_left), rsi) // i = k_left;
		test(rsi, rsi) // check i via logical AND.
		je(.SPOSTACCUM) // if i == 0, we're done; jump to end.
		 // else, we prepare to enter k_left loop.
		
		
		label(.SLOOPKLEFT) // EDGE LOOP
		
		addps(xmm6, xmm10) // iteration 0
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)
		
		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)
		
		addps(xmm2, xmm8)
		movaps(mem(rbx, -7*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)
		
		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -6*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -5*16), xmm1)
		
		sub(imm(0-1*8*4), rax) // a += 8 (1 x mr)
		sub(imm(0-1*4*4), rbx) // b += 4 (1 x nr)
		
		
		dec(rsi) // i -= 1;
		jne(.SLOOPKLEFT) // iterate again if i != 0.
		
		
		
		label(.SPOSTACCUM)
		
		addps(xmm6, xmm10)
		addps(xmm3, xmm14)
		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		
		
		mov(var(alpha), rax) // load address of alpha
		mov(var(beta), rbx) // load address of beta
		movss(mem(rax), xmm6) // load alpha to bottom 4 bytes of xmm6
		movss(mem(rbx), xmm7) // load beta to bottom 4 bytes of xmm7
		pshufd(imm(0x00), xmm6, xmm6) // populate xmm6 with four alphas
		pshufd(imm(0x00), xmm7, xmm7) // populate xmm7 with four betas
		
		
		mov(var(rs_c), rsi) // load rs_c
		mov(rsi, r8) // make a copy of rs_c
		
		lea(mem(, rsi, 4), rsi) // rsi = rs_c * sizeof(float)
		lea(mem(rsi, rsi, 2), r11) // r11 = 3*(rs_c * sizeof(float))
		
		lea(mem(rcx, rsi, 4), rdx) // load address of c + 4*rs_c;
		
		 // xmm8:   xmm9:   xmm10:  xmm11:
		 // ( ab00  ( ab01  ( ab02  ( ab03
		 //   ab11    ab12    ab13    ab10
		 //   ab22    ab23    ab20    ab21
		 //   ab33 )  ab30 )  ab31 )  ab32 )
		 //
		 // xmm12:  xmm13:  xmm14:  xmm15:
		 // ( ab40  ( ab41  ( ab42  ( ab43
		 //   ab51    ab52    ab53    ab50
		 //   ab62    ab63    ab60    ab61
		 //   ab73 )  ab70 )  ab71 )  ab72 )
		movaps(xmm9, xmm4)
		shufps(imm(0xd8), xmm8, xmm9)
		shufps(imm(0xd8), xmm11, xmm8)
		shufps(imm(0xd8), xmm10, xmm11)
		shufps(imm(0xd8), xmm4, xmm10)
		
		movaps(xmm8, xmm4)
		shufps(imm(0xd8), xmm10, xmm8)
		shufps(imm(0xd8), xmm4, xmm10)
		movaps(xmm9, xmm5)
		shufps(imm(0xd8), xmm11, xmm9)
		shufps(imm(0xd8), xmm5, xmm11)
		
		movaps(xmm13, xmm4)
		shufps(imm(0xd8), xmm12, xmm13)
		shufps(imm(0xd8), xmm15, xmm12)
		shufps(imm(0xd8), xmm14, xmm15)
		shufps(imm(0xd8), xmm4, xmm14)
		
		movaps(xmm12, xmm4)
		shufps(imm(0xd8), xmm14, xmm12)
		shufps(imm(0xd8), xmm4, xmm14)
		movaps(xmm13, xmm5)
		shufps(imm(0xd8), xmm15, xmm13)
		shufps(imm(0xd8), xmm5, xmm15)
		 // xmm8:   xmm9:   xmm10:  xmm11:
		 // ( ab00  ( ab01  ( ab02  ( ab03
		 //   ab10    ab11    ab12    ab13
		 //   ab20    ab21    ab22    ab23
		 //   ab30 )  ab31 )  ab32 )  ab33 )
		 //
		 // xmm12:  xmm13:  xmm14:  xmm15:
		 // ( ab40  ( ab41  ( ab42  ( ab43
		 //   ab50    ab51    ab52    ab53
		 //   ab60    ab61    ab62    ab63
		 //   ab70 )  ab71 )  ab72 )  ab73 )
		
		
		
		 // determine if
		 //   c      % 16 == 0, AND
		 //   8*cs_c % 16 == 0, AND
		 //   rs_c        == 1
		 // ie: aligned, ldim aligned, and
		 // column-stored
		
		cmp(imm(1), r8) // set ZF if rs_c == 1.
		sete(bl) // bl = ( ZF == 1 ? 1 : 0 );
		test(imm(15), rcx) // set ZF if c & 16 is zero.
		setz(bh) // bh = ( ZF == 1 ? 1 : 0 );
		test(imm(15), r12) // set ZF if (4*cs_c) & 16 is zero.
		setz(al) // al = ( ZF == 1 ? 1 : 0 );
		 // AND(bl,bh) followed by
		 // AND(bh,al) will reveal result
		
		 // now avoid loading C if beta == 0
		
		xorpd(xmm0, xmm0) // set xmm0 to zero.
		ucomisd(xmm0, xmm7) // check if beta == 0.
		je(.SBETAZERO) // if ZF = 1, jump to beta == 0 case
		
		
		 // check if aligned/column-stored
		AND(bl, bh) // set ZF if bl & bh == 1.
		AND(bh, al) // set ZF if bh & al == 1.
		jne(.SCOLSTORED) // jump to column storage case
		
		
		
		label(.SGENSTORED)
		
		movlps(mem(rcx), xmm0) // load c00 ~ c30
		movhps(mem(rcx, rsi, 1), xmm0)
		movlps(mem(rcx, rsi, 2), xmm1)
		movhps(mem(rcx, r11, 1), xmm1)
		shufps(imm(0x88), xmm1, xmm0)
		
		mulps(xmm6, xmm8) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm8, xmm0) // add the gemm result,
		
		movss(xmm0, mem(rcx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rcx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rcx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rcx, r11, 1))
		
		add(rdi, rcx)
		
		
		movlps(mem(rdx), xmm0) // load c40 ~ c70
		movhps(mem(rdx, rsi, 1), xmm0)
		movlps(mem(rdx, rsi, 2), xmm1)
		movhps(mem(rdx, r11, 1), xmm1)
		shufps(imm(0x88), xmm1, xmm0)
		
		mulps(xmm6, xmm12) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm12, xmm0) // add the gemm result,
		
		movss(xmm0, mem(rdx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rdx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rdx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rdx, r11, 1))
		
		add(rdi, rdx)
		
		
		movlps(mem(rcx), xmm0) // load c01 ~ c31
		movhps(mem(rcx, rsi, 1), xmm0)
		movlps(mem(rcx, rsi, 2), xmm1)
		movhps(mem(rcx, r11, 1), xmm1)
		shufps(imm(0x88), xmm1, xmm0)
		
		mulps(xmm6, xmm9) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm9, xmm0) // add the gemm result,
		
		movss(xmm0, mem(rcx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rcx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rcx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rcx, r11, 1))
		
		add(rdi, rcx)
		
		
		movlps(mem(rdx), xmm0) // load c41 ~ c71
		movhps(mem(rdx, rsi, 1), xmm0)
		movlps(mem(rdx, rsi, 2), xmm1)
		movhps(mem(rdx, r11, 1), xmm1)
		shufps(imm(0x88), xmm1, xmm0)
		
		mulps(xmm6, xmm13) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm13, xmm0) // add the gemm result,
		
		movss(xmm0, mem(rdx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rdx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rdx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rdx, r11, 1))
		
		add(rdi, rdx)
		
		
		movlps(mem(rcx), xmm0) // load c02 ~ c32
		movhps(mem(rcx, rsi, 1), xmm0)
		movlps(mem(rcx, rsi, 2), xmm1)
		movhps(mem(rcx, r11, 1), xmm1)
		shufps(imm(0x88), xmm1, xmm0)
		
		mulps(xmm6, xmm10) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm10, xmm0) // add the gemm result,
		
		movss(xmm0, mem(rcx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rcx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rcx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rcx, r11, 1))
		
		add(rdi, rcx)
		
		
		movlps(mem(rdx), xmm0) // load c42 ~ c72
		movhps(mem(rdx, rsi, 1), xmm0)
		movlps(mem(rdx, rsi, 2), xmm1)
		movhps(mem(rdx, r11, 1), xmm1)
		shufps(imm(0x88), xmm1, xmm0)
		
		mulps(xmm6, xmm14) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm14, xmm0) // add the gemm result,
		
		movss(xmm0, mem(rdx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rdx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rdx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rdx, r11, 1))
		
		add(rdi, rdx)
		
		
		movlps(mem(rcx), xmm0) // load c03 ~ c33
		movhps(mem(rcx, rsi, 1), xmm0)
		movlps(mem(rcx, rsi, 2), xmm1)
		movhps(mem(rcx, r11, 1), xmm1)
		shufps(imm(0x88), xmm1, xmm0)
		
		mulps(xmm6, xmm11) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm11, xmm0) // add the gemm result,
		
		movss(xmm0, mem(rcx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rcx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rcx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rcx, r11, 1))
		
		
		
		
		movlps(mem(rdx), xmm0) // load c43 ~ c73
		movhps(mem(rdx, rsi, 1), xmm0)
		movlps(mem(rdx, rsi, 2), xmm1)
		movhps(mem(rdx, r11, 1), xmm1)
		shufps(imm(0x88), xmm1, xmm0)
		
		mulps(xmm6, xmm15) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm15, xmm0) // add the gemm result,
		
		movss(xmm0, mem(rdx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rdx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rdx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rdx, r11, 1))
		
		
		
		
		jmp(.SDONE) // jump to end.
		
		
		
		label(.SCOLSTORED)
		
		movaps(mem(rcx), xmm0) // load c00 ~ c30,
		mulps(xmm6, xmm8) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm8, xmm0) // add the gemm result,
		movaps(xmm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx)
		
		movaps(mem(rdx), xmm1) // load c40 ~ c70,
		mulps(xmm6, xmm12) // scale by alpha,
		mulps(xmm7, xmm1) // scale by beta,
		addps(xmm12, xmm1) // add the gemm result,
		movaps(xmm1, mem(rdx)) // and store back to memory.
		add(rdi, rdx)
		
		
		
		movaps(mem(rcx), xmm0) // load c01 ~ c31,
		mulps(xmm6, xmm9) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm9, xmm0) // add the gemm result,
		movaps(xmm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx)
		
		movaps(mem(rdx), xmm1) // load c41 ~ c71,
		mulps(xmm6, xmm13) // scale by alpha,
		mulps(xmm7, xmm1) // scale by beta,
		addps(xmm13, xmm1) // add the gemm result,
		movaps(xmm1, mem(rdx)) // and store back to memory.
		add(rdi, rdx)
		
		
		
		movaps(mem(rcx), xmm0) // load c02 ~ c32,
		mulps(xmm6, xmm10) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm10, xmm0) // add the gemm result,
		movaps(xmm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx)
		
		movaps(mem(rdx), xmm1) // load c42 ~ c72,
		mulps(xmm6, xmm14) // scale by alpha,
		mulps(xmm7, xmm1) // scale by beta,
		addps(xmm14, xmm1) // add the gemm result,
		movaps(xmm1, mem(rdx)) // and store back to memory.
		add(rdi, rdx)
		
		
		
		movaps(mem(rcx), xmm0) // load c03 ~ c33,
		mulps(xmm6, xmm11) // scale by alpha,
		mulps(xmm7, xmm0) // scale by beta,
		addps(xmm11, xmm0) // add the gemm result,
		movaps(xmm0, mem(rcx)) // and store back to memory.
		
		
		movaps(mem(rdx), xmm1) // load c43 ~ c73,
		mulps(xmm6, xmm15) // scale by alpha,
		mulps(xmm7, xmm1) // scale by beta,
		addps(xmm15, xmm1) // add the gemm result,
		movaps(xmm1, mem(rdx)) // and store back to memory.
		
		jmp(.SDONE) // jump to end.
		
		
		
		
		label(.SBETAZERO)
		 // check if aligned/column-stored
		AND(bl, bh) // set ZF if bl & bh == 1.
		AND(bh, al) // set ZF if bh & al == 1.
		jne(.SCOLSTORBZ) // jump to column storage case
		
		
		
		label(.SGENSTORBZ)
		
		mulps(xmm6, xmm8) // scale by alpha,
		movaps(xmm8, xmm0)
		
		movss(xmm0, mem(rcx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rcx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rcx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rcx, r11, 1))
		
		add(rdi, rcx)
		
		
		mulps(xmm6, xmm12) // scale by alpha,
		movaps(xmm12, xmm0)
		
		movss(xmm0, mem(rdx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rdx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rdx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rdx, r11, 1))
		
		add(rdi, rdx)
		
		
		mulps(xmm6, xmm9) // scale by alpha,
		movaps(xmm9, xmm0)
		
		movss(xmm0, mem(rcx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rcx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rcx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rcx, r11, 1))
		
		add(rdi, rcx)
		
		
		mulps(xmm6, xmm13) // scale by alpha,
		movaps(xmm13, xmm0)
		
		movss(xmm0, mem(rdx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rdx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rdx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rdx, r11, 1))
		
		add(rdi, rdx)
		
		
		mulps(xmm6, xmm10) // scale by alpha,
		movaps(xmm10, xmm0)
		
		movss(xmm0, mem(rcx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rcx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rcx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rcx, r11, 1))
		
		add(rdi, rcx)
		
		
		mulps(xmm6, xmm14) // scale by alpha,
		movaps(xmm14, xmm0)
		
		movss(xmm0, mem(rdx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rdx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rdx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rdx, r11, 1))
		
		add(rdi, rdx)
		
		
		mulps(xmm6, xmm11) // scale by alpha,
		movaps(xmm11, xmm0)
		
		movss(xmm0, mem(rcx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rcx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rcx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rcx, r11, 1))
		
		
		
		
		mulps(xmm6, xmm15) // scale by alpha,
		movaps(xmm15, xmm0)
		
		movss(xmm0, mem(rdx)) // and store back to memory.
		pshufd(imm(0x39), xmm0, xmm1)
		movss(xmm1, mem(rdx, rsi, 1))
		pshufd(imm(0x39), xmm1, xmm2)
		movss(xmm2, mem(rdx, rsi, 2))
		pshufd(imm(0x39), xmm2, xmm3)
		movss(xmm3, mem(rdx, r11, 1))
		
		
		
		
		jmp(.SDONE) // jump to end.
		
		
		
		label(.SCOLSTORBZ)
		
		 // skip loading c00 ~ c30,
		mulps(xmm6, xmm8) // scale by alpha,
		movaps(xmm8, mem(rcx)) // and store back to memory.
		add(rdi, rcx)
		 // skip loading c40 ~ c70,
		mulps(xmm6, xmm12) // scale by alpha,
		movaps(xmm12, mem(rdx)) // and store back to memory.
		add(rdi, rdx)
		
		
		 // skip loading c01 ~ c31,
		mulps(xmm6, xmm9) // scale by alpha,
		movaps(xmm9, mem(rcx)) // and store back to memory.
		add(rdi, rcx)
		 // skip loading c41 ~ c71,
		mulps(xmm6, xmm13) // scale by alpha,
		movaps(xmm13, mem(rdx)) // and store back to memory.
		add(rdi, rdx)
		
		
		 // skip loading c02 ~ c32,
		mulps(xmm6, xmm10) // scale by alpha,
		movaps(xmm10, mem(rcx)) // and store back to memory.
		add(rdi, rcx)
		 // skip loading c42 ~ c72,
		mulps(xmm6, xmm14) // scale by alpha,
		movaps(xmm14, mem(rdx)) // and store back to memory.
		add(rdi, rdx)
		
		
		 // skip loading c03 ~ c33,
		mulps(xmm6, xmm11) // scale by alpha,
		movaps(xmm11, mem(rcx)) // and store back to memory.
		
		 // skip loading c43 ~ c73,
		mulps(xmm6, xmm15) // scale by alpha,
		movaps(xmm15, mem(rdx)) // and store back to memory.
		
		
		
		
		
		
		
		
		label(.SDONE)
		

    end_asm(
		: // output operands (none)
		: // input operands
	      [k_iter] "m" (k_iter), // 0
	      [k_left] "m" (k_left), // 1
	      [a]      "m" (a),      // 2
	      [b]      "m" (b),      // 3
	      [alpha]  "m" (alpha),  // 4
	      [beta]   "m" (beta),   // 5
	      [c]      "m" (c),      // 6
	      [rs_c]   "m" (rs_c),   // 7
	      [cs_c]   "m" (cs_c),   // 8
	      [b_next] "m" (b_next)/*, // 9
	      [a_next] "m" (a_next)*/  // 10
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	)
}

template<>
void GEMM<float>::Kernel(
    int64_t k,
    float *alpha,
    float *a,
    float *b,
    float *beta,
    float *c, int64_t rs_c, int64_t cs_c,
    const float *, const float *b_next) {
  bli_sgemm_penryn_asm_8x4(k, alpha, a, b, beta, c, rs_c, cs_c, b_next);
}



}  // namespace pocketkaldi