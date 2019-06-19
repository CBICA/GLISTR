///////////////////////////////////////////////////////////////////////////////////////
// SMatrix.cpp
// Developed by Dongjin Kwon
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2011-2014 Dongjin Kwon
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include <math.h>
#include "SMatrix.h"
#include "MyUtils.h"


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CSMatrix::CSMatrix()
{
	m_val = NULL;
	m_rowind = NULL;
	m_colind = NULL;
	m_nz = 0;
	m_nz_max = 0;
	m_rows = m_cols = 0;

	m_cc_val = NULL;
	m_cc_rowind = NULL;
	m_cc_colptr = NULL;
	m_cr_val = NULL;
	m_cr_rowptr = NULL;
	m_cr_colind = NULL;

	m_indx = NULL;
	m_LU = NULL;
}

CSMatrix::CSMatrix(int rows, int cols)
{
	m_val = NULL;
	m_rowind = NULL;
	m_colind = NULL;
	m_nz = 0;
	m_nz_max = 0;
	m_rows = m_cols = 0;

	m_cc_val = NULL;
	m_cc_rowind = NULL;
	m_cc_colptr = NULL;
	m_cr_val = NULL;
	m_cr_rowptr = NULL;
	m_cr_colind = NULL;

	m_indx = NULL;
	m_LU = NULL;

	Init(rows, cols);
}

CSMatrix::CSMatrix(int rows, int cols, int nz_max)
{
	m_val = NULL;
	m_rowind = NULL;
	m_colind = NULL;
	m_nz = 0;
	m_nz_max = 0;
	m_rows = m_cols = 0;

	m_cc_val = NULL;
	m_cc_rowind = NULL;
	m_cc_colptr = NULL;
	m_cr_val = NULL;
	m_cr_rowptr = NULL;
	m_cr_colind = NULL;

	m_indx = NULL;
	m_LU = NULL;

	Init(rows, cols, nz_max);
}

CSMatrix::~CSMatrix()
{
	Free();
}

void CSMatrix::Init(int rows, int cols)
{
	Free();

	m_rows = rows;
	m_cols = cols;
	m_nz_max = (int)(300*MAX(rows, cols));
	m_val = (double*)MyAllocEx(m_nz_max*sizeof(double), "CSMatrix::Init m_val");
	m_rowind = (int*)MyAllocEx(m_nz_max*sizeof(int), "CSMatrix::Init m_rowind");
	m_colind = (int*)MyAllocEx(m_nz_max*sizeof(int), "CSMatrix::Init m_colind");
}

void CSMatrix::Init(int rows, int cols, int nz_max)
{
	Free();

	m_rows = rows;
	m_cols = cols;
	m_nz_max = nz_max;
	m_val = (double*)MyAllocEx(m_nz_max*sizeof(double), "CSMatrix::Init m_val");
	m_rowind = (int*)MyAllocEx(m_nz_max*sizeof(int), "CSMatrix::Init m_rowind");
	m_colind = (int*)MyAllocEx(m_nz_max*sizeof(int), "CSMatrix::Init m_colind");
}

void CSMatrix::Free()
{
	if (m_val)       { MyFree(m_val); m_val = NULL; }
	if (m_rowind)    { MyFree(m_rowind); m_rowind = NULL; }
	if (m_colind)    { MyFree(m_colind); m_colind = NULL; }
	if (m_cc_val)    { MyFree(m_cc_val); m_cc_val = NULL; }
	if (m_cc_rowind) { MyFree(m_cc_rowind); m_cc_rowind = NULL; }
	if (m_cc_colptr) { MyFree(m_cc_colptr); m_cc_colptr = NULL; }
	if (m_cr_val)    { MyFree(m_cr_val); m_cr_val = NULL; }
	if (m_cr_rowptr) { MyFree(m_cr_rowptr); m_cr_rowptr = NULL; }
	if (m_cr_colind) { MyFree(m_cr_colind); m_cr_colind = NULL; }
	//
	if (m_indx)      { MyFree(m_indx); m_indx = NULL; }
	if (m_LU)		 { delete m_LU; m_LU = NULL; }
	//
	m_nz = 0;
	m_nz_max = 0;
	m_rows = m_cols = 0;
}

void CSMatrix::Add(int r, int c, double val)
{
//#ifdef _DEBUG
#if 0
	if ((r < 0) || (r >= m_rows)) {
		AfxMessageBox("(r < 0) || (r >= m_rows)");
		return;
	}
	if ((c < 0) || (c > m_cols)) {
		AfxMessageBox("(c < 0) || (c > m_cols)");
		return;
	}
	if (m_nz >= m_nz_max) {
		AfxMessageBox("m_nz reaches to m_nz_max");
		return;
	}
#endif
	m_val[m_nz] = val;
	m_rowind[m_nz] = r;
	m_colind[m_nz] = c;
	m_nz++;
}

void CSMatrix::Set(double alpha, CDMatrix& X)
{
	if (m_val != NULL) {
		if ((m_rows == X.m_rows) && (m_cols == X.m_cols)) {
			m_nz = 0;
		} else {
			Init(X.m_rows, X.m_cols);
		}
	} else {
		Init(X.m_rows, X.m_cols);
	}

	int i, j;
	for (j = 0; j < m_rows; j++) {
		double* p_val = X.m_dm[j];
		for (i = 0; i < m_cols; i++) {
			double val = *p_val++;
			if (val != 0) {
				Add(j, i, alpha * val);
			}
		}
	}
}

void CSMatrix::GenerateCompInfo()
{
	if (m_nz == 0) { return; }

	if (m_cc_val)    { MyFree(m_cc_val); m_cc_val = NULL; }
	if (m_cc_rowind) { MyFree(m_cc_rowind); m_cc_rowind = NULL; }
	if (m_cc_colptr) { MyFree(m_cc_colptr); m_cc_colptr = NULL; }
	if (m_cr_val)    { MyFree(m_cr_val); m_cr_val = NULL; }
	if (m_cr_rowptr) { MyFree(m_cr_rowptr); m_cr_rowptr = NULL; }
	if (m_cr_colind) { MyFree(m_cr_colind); m_cr_colind = NULL; }

	m_cc_val    = (double*)MyAllocEx(m_nz*sizeof(double), "CSMatrix::GenerateCompInfo m_cc_val");
	m_cc_rowind = (int*)MyAllocEx(m_nz*sizeof(int), "CSMatrix::GenerateCompInfo m_cc_rowind");
	m_cc_colptr = (int*)MyAllocEx((m_cols+1)*sizeof(int), "CSMatrix::GenerateCompInfo m_cc_colptr");
	m_cr_val    = (double*)MyAllocEx(m_nz*sizeof(double), "CSMatrix::GenerateCompInfo m_cr_val");
	m_cr_rowptr = (int*)MyAllocEx((m_rows+1)*sizeof(int), "CSMatrix::GenerateCompInfo m_cr_rowptr");
	m_cr_colind = (int*)MyAllocEx(m_nz*sizeof(int), "CSMatrix::GenerateCompInfo m_cr_colind");

	int i, j, r, c, idx;
	int* row_buf;
	int* col_buf;
	double val;

	row_buf = (int*)MyAllocEx(m_rows*sizeof(int), "CSMatrix::GenerateCompInfo row_buf");
	col_buf = (int*)MyAllocEx(m_cols*sizeof(int), "CSMatrix::GenerateCompInfo col_buf");
	memset(row_buf, 0, m_rows*sizeof(int));
	memset(col_buf, 0, m_cols*sizeof(int));

	for (j = 0; j < m_nz; j++) {
		row_buf[m_rowind[j]]++;
		col_buf[m_colind[j]]++;
	}
	m_cr_rowptr[0] = 0;
	for (i = 1; i <= m_rows; i++) {
		m_cr_rowptr[i] = m_cr_rowptr[i-1] + row_buf[i-1];
	}
	m_cc_colptr[0] = 0;
	for (i = 1; i <= m_cols; i++) {
		m_cc_colptr[i] = m_cc_colptr[i-1] + col_buf[i-1];
	}
	for (j = 0; j < m_nz; j++) {
		r = m_rowind[j];
		c = m_colind[j];
		val = m_val[j];
		row_buf[r]--;
		idx = m_cr_rowptr[r]+row_buf[r];
		m_cr_colind[idx] = c;
		m_cr_val[idx] = val;
		col_buf[c]--;
		idx = m_cc_colptr[c]+col_buf[c];
		m_cc_rowind[idx] = r;
		m_cc_val[idx] = val;
	}

	MyFree(row_buf);
	MyFree(col_buf);
}

void CSMatrix::Mult(double s)
{
	int j;
	double* p_val = m_val;
	for (j = 0; j < m_nz; j++) {
		(*p_val) = s * (*p_val);
		p_val++;
	}
}

void CSMatrix::Set_APB(double alpha, CSMatrix& A, CSMatrix& B)
{
	if (A.m_cc_val == NULL) {
		A.GenerateCompInfo();
	}
	if (B.m_cr_val == NULL) {
		B.GenerateCompInfo();
	}

	int i, j, k;
	int i_cp1, i_cp2, j_cp1, j_cp2;

	CDMatrix dm_C(A.m_rows, B.m_cols);

	if (alpha == 1.0) {
		for (k = 0; k < A.m_cols; k++) {
			j_cp1 = A.m_cc_colptr[k];
			j_cp2 = A.m_cc_colptr[k+1];
			i_cp1 = B.m_cr_rowptr[k];
			i_cp2 = B.m_cr_rowptr[k+1];
			for (j = j_cp1; j < j_cp2; j++) {
				for (i = i_cp1; i < i_cp2; i++) {
					dm_C.m_dm[A.m_cc_rowind[j]][B.m_cr_colind[i]] += A.m_cc_val[j] * B.m_cr_val[i];
				}
			}
		}
	} else {
		for (k = 0; k < A.m_cols; k++) {
			j_cp1 = A.m_cc_colptr[k];
			j_cp2 = A.m_cc_colptr[k+1];
			i_cp1 = B.m_cr_rowptr[k];
			i_cp2 = B.m_cr_rowptr[k+1];
			for (j = j_cp1; j < j_cp2; j++) {
				for (i = i_cp1; i < i_cp2; i++) {
					dm_C.m_dm[A.m_cc_rowind[j]][B.m_cr_colind[i]] += alpha * A.m_cc_val[j] * B.m_cr_val[i];
				}
			}
		}
	}

//	dm_C.SaveMatrix("dm_C.txt", 0);
//	dm_C.SaveMatrix("dm_C_spy.txt", 1);

	Set(1.0, dm_C);
}

//*
void CSMatrix::Set_ATA(double alpha, CSMatrix& A)
{
	if (A.m_cr_val == NULL) {
		A.GenerateCompInfo();
	}

	int i, j, k;
	int cp1, cp2;

	// the size of C is A.m_cols by A.m_cols
	CDMatrix dm_C(A.m_cols, A.m_cols);

	if (alpha == 1.0) {
		for (k = 0; k < A.m_rows; k++) {
			cp1 = A.m_cr_rowptr[k];
			cp2 = A.m_cr_rowptr[k+1];
			for (j = cp1; j < cp2; j++) {
				for (i = cp1; i < cp2; i++) {
					dm_C.m_dm[A.m_cr_colind[j]][A.m_cr_colind[i]] += A.m_cr_val[j] * A.m_cr_val[i];
				}
			}
		}
	} else {
		for (k = 0; k < A.m_rows; k++) {
			cp1 = A.m_cr_rowptr[k];
			cp2 = A.m_cr_rowptr[k+1];
			for (j = cp1; j < cp2; j++) {
				for (i = cp1; i < cp2; i++) {
					dm_C.m_dm[A.m_cr_colind[j]][A.m_cr_colind[i]] += alpha * A.m_cr_val[j] * A.m_cr_val[i];
				}
			}
		}
	}

//	dm_C.SaveMatrix("dm_C.txt", 0);
//	dm_C.SaveMatrix("dm_C_spy.txt", 1);

	Set(1.0, dm_C);
}
/*/
void CSMatrix::Set_ATA(double alpha, CSMatrix& A)
{
	if (A.m_cr_val == NULL) {
		A.GenerateCompInfo();
	}

	int i, j, k;
	int cp1, cp2;

	if (m_val != NULL) {
		if ((m_rows == A.m_cols) && (m_cols == A.m_cols)) {
			m_nz = 0;
		} else {
			Init(A.m_cols, A.m_cols);
		}
	}

	int rows = m_cols;
	int* row_nz = (int*)MyAlloc(rows * sizeof(int));
	double** val = (double**)MyAlloc(rows * sizeof(double*));
	int** colind = (int**)MyAlloc(rows * sizeof(int*));
	for (j = 0; j < rows; j++) {
		val[j] = (double*)MyAlloc(SM_CMAX * sizeof(double));
		colind[j] = (int*)MyAlloc(SM_CMAX * sizeof(int));
		row_nz[j] = 0;
	}
	int rind, cind, rnz;

	if (alpha == 1.0) {
		for (k = 0; k < A.m_rows; k++) {
			cp1 = A.m_cr_rowptr[k];
			cp2 = A.m_cr_rowptr[k+1];
			for (j = cp1; j < cp2; j++) {
				for (i = cp1; i < cp2; i++) {
					//if (row_nz[rind] >= SM_CMAX-1) {
					//	AfxMessageBox("Set_ATA: row_nz[rind] >= SM_CMAX-1");
					//}
					rind = A.m_cr_colind[j];
					cind = A.m_cr_colind[i];
					for (rnz = 0; rnz < row_nz[rind]; rnz++) {
						if (colind[rind][rnz] == cind) {
							val[rind][rnz] += A.m_cr_val[j] * A.m_cr_val[i];
							break;
						}
					}
					if (rnz == row_nz[rind]) {
						val[rind][rnz] = A.m_cr_val[j] * A.m_cr_val[i];
						colind[rind][rnz] = cind;
						row_nz[rind]++;
					}
				}
			}
		}
	} else {
		for (k = 0; k < A.m_rows; k++) {
			cp1 = A.m_cr_rowptr[k];
			cp2 = A.m_cr_rowptr[k+1];
			for (j = cp1; j < cp2; j++) {
				for (i = cp1; i < cp2; i++) {
					//if (row_nz[rind] >= SM_CMAX-1) {
					//	AfxMessageBox("Set_ATA: row_nz[rind] >= SM_CMAX-1");
					//}
					rind = A.m_cr_colind[j];
					cind = A.m_cr_colind[i];
					for (rnz = 0; rnz < row_nz[rind]; rnz++) {
						if (colind[rind][rnz] == cind) {
							val[rind][rnz] += alpha * A.m_cr_val[j] * A.m_cr_val[i];
							break;
						}
					}
					if (rnz == row_nz[rind]) {
						val[rind][rnz] = alpha * A.m_cr_val[j] * A.m_cr_val[i];
						colind[rind][rnz] = cind;
						row_nz[rind]++;
					}
				}
			}
		}
	}

	{
		m_nz = 0;
		double* p_m_val = m_val;
		int* p_m_rowind = m_rowind;
		int* p_m_colind = m_colind;
		for (rind = 0; rind < rows; rind++) {
			double* p_v = val[rind];
			int* p_c = colind[rind];
			double v;
			for (i = 0; i < row_nz[rind]; i++) {
				v = *p_v++;
				if (v != 0) {
					*p_m_val++ = v;
					*p_m_rowind++ = rind;
					*p_m_colind++ = *p_c;
					m_nz++;
				}
				p_c++;
			}
		}
	}

	for (j = 0; j < rows; j++) {
		MyFree(val[j]);
		MyFree(colind[j]);
	}
	MyFree(val);
	MyFree(colind);
	MyFree(row_nz);
}
//*/
/*
	{
		m_nz = 0;
		double* p_m_val = m_val;
		int* p_m_rowind = m_rowind;
		int* p_m_colind = m_colind;
		for (rind = 0; rind < rows; rind++) {
			double* p_v = val[rind];
			int* p_c = colind[rind];
			for (i = 0; i < row_nz[rind]; i++) {
				if (*p_v == 0) {
					AfxMessageBox("Set_ATA: v = 0");
				}
				*p_m_val++ = *p_v++;
				*p_m_rowind++ = rind;
				*p_m_colind++ = *p_c++;
				m_nz++;
			}
		}
	}
*/
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Invoke the < comparison function
#define CMP(A,B) ((A) < (B))

// swap two items
//
static inline void SWAP_double(double &A, double &B)
{
    double tmp = A; A = B; B = tmp;
}

static inline void swap_int(int &a, int &b)
{
    int tmp=a; a=b; b=tmp;
}

// This should be replaced by a standard ANSI macro.
#define BYTES_PER_WORD 8

/* The next 4 #defines implement a very fast in-line stack abstraction. */
#define STACK_SIZE (BYTES_PER_WORD * sizeof (long))
#define PUSH(LOW,HIGH) do {top->lo = LOW;top++->hi = HIGH;} while (0)
#define POP(LOW,HIGH)  do {LOW = (--top)->lo;HIGH = top->hi;} while (0)
#define STACK_NOT_EMPTY (stack < top)                

/* Discontinue quicksort algorithm when partition gets below this size.
   This particular magic number was chosen to work best on a Sun 4/260. */
#define MAX_THRESH 4

/* Stack node declarations used to store unfulfilled partition obligations. */
typedef struct {
  int lo;
  int hi;
} stack_node;

/* Order size using quicksort.  This implementation incorporates
   four optimizations discussed in Sedgewick:
   
   1. Non-recursive, using an explicit stack of pointer that store the 
      next array partition to sort.  To save time, this maximum amount 
      of space required to store an array of MAX_INT is allocated on the 
      stack.  Assuming a 32-bit integer, this needs only 32 * 
      sizeof (stack_node) == 136 bits.  Pretty cheap, actually.

   2. Chose the pivot element using a median-of-three decision tree.
      This reduces the probability of selecting a bad pivot value and 
      eliminates certain extraneous comparisons.

   3. Only quicksorts TOTAL_ELEMS / MAX_THRESH partitions, leaving
      insertion sort to order the MAX_THRESH items within each partition.  
      This is a big win, since insertion sort is faster for small, mostly
      sorted array segments.
   
   4. The larger of the two sub-partitions is always pushed onto the
      stack first, with the algorithm then concentrating on the
      smaller partition.  This *guarantees* no more than log (n)
      stack size is needed (actually O(1) in this case)! */

// example
//  QSort(l_rowind_, l_val_, l_colptr_[i], l_colptr_[i+1] - l_colptr_[i]);
//  QSort(u_rowind_, u_val_, u_colptr_[i], u_colptr_[i+1] - u_colptr_[i]);

int QSort(int* v, double* x, int base_ptr, int total_elems)
{
	int pivot_buffer;
	double pixot_buffer;
	
	if (total_elems > MAX_THRESH) {
		
		int lo = base_ptr;
		int hi = lo + total_elems - 1;
		
		stack_node stack[STACK_SIZE]; /* Largest size needed for 32-bit int!!! */
		stack_node *top = stack + 1;
		
		while (STACK_NOT_EMPTY) {
			int left_ptr;
			int right_ptr;
			{
				{
					/* Select median value from among LO, MID, and HI. Rearrange
					LO and HI so the three values are sorted. This lowers the 
					probability of picking a pathological pivot value and 
					skips a comparison for both the LEFT_PTR and RIGHT_PTR. */
					
					int mid = lo + (hi - lo) / 2;
					
					if (CMP (v[mid], v[lo])) {
						swap_int (v[mid], v[lo]);
						SWAP_double (x[mid], x[lo]);
					}
					if (CMP (v[hi], v[mid])) {
						swap_int (v[hi], v[mid]);
						SWAP_double (x[hi], x[mid]);
					} else 
						goto jump_over;
					
					if (CMP (v[mid], v[lo])) {
						swap_int(v[mid], v[lo]);
						SWAP_double (x[mid], x[lo]);
					}
					
jump_over:
					
					pivot_buffer = v[mid];
					pixot_buffer = x[mid];
				}
				
				left_ptr  = lo + 1;
				right_ptr = hi - 1;
				
				/* Here's the famous ``collapse the walls'' section of quicksort.  
				Gotta like those tight inner loops!  They are the main reason 
				that this algorithm runs much faster than others. */
				do {
					while (CMP (v[left_ptr], pivot_buffer))
						left_ptr++;
					
					while (CMP (pivot_buffer, v[right_ptr]))
						right_ptr--;
					
					if (left_ptr < right_ptr) {
						swap_int (v[left_ptr], v[right_ptr]);
						SWAP_double (x[left_ptr], x[right_ptr]);
						left_ptr++;
						right_ptr--;
					} else if (left_ptr == right_ptr) {
						left_ptr ++;
						right_ptr --;
						break;
					}
				} while (left_ptr <= right_ptr);
			}
			
			/* Set up pointers for next iteration.  First determine whether
			left and right partitions are below the threshold size. If so, 
			ignore one or both.  Otherwise, push the larger partition's
			bounds on the stack and continue sorting the smaller one. */
			
			if ((right_ptr - lo) <= MAX_THRESH) {
				if ((hi - left_ptr) <= MAX_THRESH)
					POP (lo, hi); 
				else
					lo = left_ptr;
			} else if ((hi - left_ptr) <= MAX_THRESH)
				hi = right_ptr;
			else if ((right_ptr - lo) > (hi - left_ptr)) {                   
				PUSH (lo, right_ptr);
				lo = left_ptr;
			} else {                   
				PUSH (left_ptr, hi);
				hi = right_ptr;
			}
		}
	}
  
	/* Once the BASE_PTR array is partially sorted by quicksort the rest
	is completely sorted using insertion sort, since this is efficient 
	for partitions below MAX_THRESH size. BASE_PTR points to the beginning 
	of the array to sort, and END_PTR points at the very last element in
	the array (*not* one beyond it!). */
	
#define QSort_MIN(X,Y) ((X) < (Y) ? (X) : (Y))
	
	{
		int end_ptr = base_ptr + total_elems - 1;
		int run_ptr;
		int tmp_ptr = base_ptr;
		int thresh = QSort_MIN(end_ptr, base_ptr + MAX_THRESH);
		
		for (run_ptr = tmp_ptr + 1; run_ptr <= thresh; run_ptr++) {
			if (CMP (v[run_ptr], v[tmp_ptr])) {
				tmp_ptr = run_ptr;
			}
		}
		
		if (tmp_ptr != base_ptr) {
			swap_int(v[tmp_ptr], v[base_ptr]);
			SWAP_double (x[tmp_ptr], x[base_ptr]);
		}
		
		for (run_ptr = base_ptr + 1; (tmp_ptr = run_ptr += 1) <= end_ptr;) {

			while (CMP (v[run_ptr], v[tmp_ptr -= 1]))
				;
			
			if ((tmp_ptr += 1) != run_ptr) {
				int trav;
				
				for (trav = run_ptr + 1; --trav >= run_ptr;) {
					int  c;
					double d;
					c = v[trav];
					d = x[trav];
					int hi, lo;
					
					for (hi = lo = trav; (lo -= 1) >= tmp_ptr; hi = lo) {
						v[hi] = v[lo];
						x[hi] = x[lo];
					}
					v[hi] = c;
					x[hi] = d;
				}
			}
		}
	}
	
	return 1;
}
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
#define TINY 1e-20
void CSMatrix::GenerateLU()
{
	if (m_indx) { MyFree(m_indx); m_indx = NULL; }
	if (m_LU)	{ delete m_LU; m_LU = NULL; }

	int i, imax, j, k;
	double big, dum, sum, temp;
	double *vv;
	double d;

	m_indx = (int*)MyAlloc(m_rows*sizeof(int));
	m_LU = new CDMatrix(m_rows, m_cols, this, FALSE);

	vv = (double*)MyAlloc(m_rows*sizeof(double));
	d = 1.0;
	for (i = 0; i < m_rows; i++) {
		big = 0.0;

		for (j = m_cr_rowptr[i]; j < m_cr_rowptr[i+1]; j++) {
			if ((temp = m_cr_val[j]) > big) big = temp;
		}
		if (big == 0.0) {
			AfxMessageBox("Singular matrix in routine ludcmp\r\n");
			return;
		}
		vv[i] = 1.0 / big;
	}
	for (j = 0; j < m_rows; j++) {
		for (i = 0; i < j; i++) {
			sum = m_LU->m_dm[i][j];
			for (k = 0; k < i; k++) sum -= m_LU->m_dm[i][k]*m_LU->m_dm[k][j];
			m_LU->m_dm[i][j] = sum;
		}
		big = 0.0;
		imax = 0;
		for (i = j; i < m_rows; i++) {
			sum = m_LU->m_dm[i][j];
			for (k = 0; k < j; k++) sum -= m_LU->m_dm[i][k]*m_LU->m_dm[k][j];
			m_LU->m_dm[i][j] = sum;
			if ((dum = vv[i]*fabs(sum)) >= big) {
				big = dum;
				imax = i;
			}
		}
		if (j != imax) {
			for (k = 0; k < m_rows; k++) {
				dum = m_LU->m_dm[imax][k];
				m_LU->m_dm[imax][k] = m_LU->m_dm[j][k];
				m_LU->m_dm[j][k] = dum;
			}
			d = -d;
			vv[imax] = vv[j];
		}
		m_indx[j] = imax;
		if (m_LU->m_dm[j][j] == 0.0) m_LU->m_dm[j][j] = TINY;
		if (j != m_rows-1) {
			dum = 1.0 / (m_LU->m_dm[j][j]);
			for (i = j+1; i < m_rows; i++) m_LU->m_dm[i][j] *= dum;
		}
	}
	MyFree(vv);
}

void CSMatrix::SolveLU(CDVector& b, CDVector& x)
{
	int i, ii = 0, ip, j;
	double sum;

	x.Set(b);

	for (i = 0; i < m_rows; i++) {
		ip = m_indx[i];
		sum = x.m_dv[ip];
		x.m_dv[ip] = x.m_dv[i];
		if (ii != 0) {
			for (j = ii-1; j < i; j++) sum -= m_LU->m_dm[i][j]*x.m_dv[j];
		} else if (sum != 0.0) {
			ii = i+1;
		}
		x.m_dv[i] = sum;
	}
	for (i = m_rows-1; i >= 0; i--) {
		sum = x.m_dv[i];
		for (j = i+1; j < m_rows; j++) sum -= m_LU->m_dm[i][j]*x.m_dv[j];
		x.m_dv[i] = sum / m_LU->m_dm[i][i];
	}

/*
	CDMatrix A(m_rows, m_cols, this, FALSE);
	int i;
	for (i = 0; i < m_rows; i++) {
		x.m_dv[i] = b.m_dv[i] / A.m_dm[i][i];
	}
*/
}

#if 1
//#define USE_UMFPACK
#define USE_SUPERLU_MT
//#define USE_SUPERLU

//#include "mkl.h"
#ifdef USE_UMFPACK
#include "umfpack.h"
#endif
#ifdef USE_SUPERLU_MT
#undef MAX
#undef MIN
#include "pdsp_defs.h"
#endif
#ifdef USE_SUPERLU
#include "slu_ddefs.h"
#endif

#ifdef USE_UMFPACK
static
void ReportError(LPCTSTR str_func, UF_long error_code) {
	char str_tmp[2048];
	sprintf(str_tmp, "%s: error_code = %d\n", str_func, error_code);
	AfxMessageBox(str_tmp);
}
#endif

BOOL CSMatrix::SolveLinearSystem(CDVector& b, CDVector& x)
{
#ifdef USE_UMFPACK
#if defined(WIN32) && !defined(WIN64)
    double Info[UMFPACK_INFO], Control[UMFPACK_CONTROL];
    void *Symbolic, *Numeric;
	int status;
	int *Ap, *Ai;
	double *Ax;

    Ap = (int*)MyAllocEx((m_cols+1)*sizeof(int), "CSMatrix::SolveLinearSystem Ap");
    Ai = (int*)MyAllocEx(m_nz*sizeof(int), "CSMatrix::SolveLinearSystem Ai");
    Ax = (double*)MyAllocEx(m_nz*sizeof(double), "CSMatrix::SolveLinearSystem Ax");

    umfpack_di_defaults(Control);
	status = umfpack_di_triplet_to_col (m_rows, m_cols, m_nz, m_rowind, m_colind, m_val, Ap, Ai, Ax, (int*)NULL);
    if (status < 0) { ReportError("umfpack_di_triplet_to_col", status); return FALSE; }
	status = umfpack_di_symbolic(m_rows, m_cols, Ap, Ai, Ax, &Symbolic, Control, Info);
    if (status < 0) { ReportError("umfpack_di_symbolic", status); return FALSE; }
    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_di_numeric", status); return FALSE; }
    
	status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x.m_dv, b.m_dv, Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_di_solve", status); return FALSE; }

    umfpack_di_free_numeric(&Numeric);
    umfpack_di_free_symbolic(&Symbolic);

	MyFree(Ap);
	MyFree(Ai);
	MyFree(Ax);
#else
    double Info[UMFPACK_INFO], Control[UMFPACK_CONTROL];
    void *Symbolic, *Numeric;
	UF_long status;
	UF_long *Ap, *Ai;
	double *Ax;
	UF_long* rowind;
	UF_long* colind;
	int i;

    Ap = (UF_long*)MyAllocEx((m_cols+1)*sizeof(UF_long), "CSMatrix::SolveLinearSystem Ap");
    Ai = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem Ai");
    Ax = (double*)MyAllocEx(m_nz*sizeof(double), "CSMatrix::SolveLinearSystem Ax");
	rowind = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem rowind");
	colind = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem colind");

	for (i = 0; i < m_nz; i++) {
		rowind[i] = (UF_long)m_rowind[i];
		colind[i] = (UF_long)m_colind[i];
	}

    umfpack_dl_defaults(Control);
	status = umfpack_dl_triplet_to_col (m_rows, m_cols, m_nz, rowind, colind, m_val, Ap, Ai, Ax, (UF_long*)NULL);
    if (status < 0) { ReportError("umfpack_dl_triplet_to_col", status); return FALSE; }
	status = umfpack_dl_symbolic(m_rows, m_cols, Ap, Ai, Ax, &Symbolic, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_symbolic", status); return FALSE; }
    status = umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_numeric", status); return FALSE; }
    
	status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x.m_dv, b.m_dv, Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_solve", status); return FALSE; }

    umfpack_dl_free_numeric(&Numeric);
    umfpack_dl_free_symbolic(&Symbolic);

	MyFree(Ap);
	MyFree(Ai);
	MyFree(Ax);
	MyFree(rowind);
	MyFree(colind);
#endif
#endif
#ifdef USE_SUPERLU_MT
	SuperMatrix A;
    NCformat *Astore;
    int *perm_r; // row permutations from partial pivoting
    int *perm_c; // column permutation vector
    SuperMatrix L; // factor L
    SCPformat *Lstore;
    SuperMatrix U; // factor U
    NCPformat *Ustore;
    SuperMatrix B;
	int nrhs, info;
	int nprocs; // maximum number of processors to use.
	int panel_size, relax, maxsup;
	int permc_spec;
	double *rhs;
	superlu_memusage_t superlu_memusage;
	int i;

	if (m_cc_colptr == NULL) {
		GenerateCompInfo();
	}

#ifdef _OPENMP
	#pragma omp parallel
	{
		nprocs = omp_get_num_threads();
	}
#else
	nprocs = 1;
#endif
	TRACE2("nprocs = %d\n", nprocs);
	//
	nrhs = 1;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    maxsup = sp_ienv(3);

    dCreate_CompCol_Matrix(&A, m_rows, m_cols, m_nz, m_cc_val, m_cc_rowind, m_cc_colptr, SLU_NC, SLU_D, SLU_GE);
    Astore = (NCformat*)A.Store;
    TRACE2("Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);
    
    if (!(rhs = doubleMalloc(m_rows * nrhs))) {
		TRACE("Malloc fails for rhs[].");
		return FALSE;
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i] = b.m_dv[i];
	}

    dCreate_Dense_Matrix(&B, m_rows, nrhs, rhs, m_rows, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m_rows))) {
		TRACE("Malloc fails for perm_r[].");
		return FALSE;
	}
    if (!(perm_c = intMalloc(m_cols))) {
		TRACE("Malloc fails for perm_c[].");
		return FALSE;
	}

    // Get column permutation vector perm_c[], according to permc_spec:
    //   permc_spec = 0: natural ordering 
    //   permc_spec = 1: minimum degree ordering on structure of A'*A
    //   permc_spec = 2: minimum degree ordering on structure of A'+A
    //   permc_spec = 3: approximate minimum degree for unsymmetric matrices
    permc_spec = 3;
    get_perm_c(permc_spec, &A, perm_c);

	pdgssv(nprocs, &A, perm_c, perm_r, &L, &U, &B, &info);

    if (info == 0) {
		for (i = 0; i < m_rows; i++) {
			x.m_dv[i] = rhs[i];
		}

		Lstore = (SCPformat*)L.Store;
		Ustore = (NCPformat*)U.Store;
    	TRACE2("#NZ in factor L = %d\n", Lstore->nnz);
    	TRACE2("#NZ in factor U = %d\n", Ustore->nnz);
    	TRACE2("#NZ in L+U = %d\n", Lstore->nnz + Ustore->nnz - L.ncol);
	
		superlu_dQuerySpace(nprocs, &L, &U, panel_size, &superlu_memusage);
		TRACE2("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
			superlu_memusage.for_lu/1024/1024, 
			superlu_memusage.total_needed/1024/1024,
			superlu_memusage.expansions);
    }

    SUPERLU_FREE (rhs);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
    Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_SCP(&L);
    Destroy_CompCol_NCP(&U);

	if (info != 0) {
		return FALSE;
	}
#endif
#ifdef USE_SUPERLU
	SuperMatrix A;
    NCformat *Astore;
    int *perm_r; // row permutations from partial pivoting
    int *perm_c; // column permutation vector
    SuperMatrix L; // factor L
    SCformat *Lstore;
    SuperMatrix U; // factor U
    NCformat *Ustore;
    SuperMatrix B;
	int nrhs, info;
	double *rhs;
    mem_usage_t mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;
	int i;

	if (m_cc_colptr == NULL) {
		GenerateCompInfo();
	}

	set_default_options(&options);

	nrhs = 1;

    dCreate_CompCol_Matrix(&A, m_rows, m_cols, m_nz, m_cc_val, m_cc_rowind, m_cc_colptr, SLU_NC, SLU_D, SLU_GE);
    Astore = (NCformat*)A.Store;
    TRACE2("Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);
    
    if (!(rhs = doubleMalloc(m_rows * nrhs))) {
		TRACE("Malloc fails for rhs[].");
		return FALSE;
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i] = b.m_dv[i];
	}

    dCreate_Dense_Matrix(&B, m_rows, nrhs, rhs, m_rows, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m_rows))) {
		TRACE("Malloc fails for perm_r[].");
		return FALSE;
	}
    if (!(perm_c = intMalloc(m_cols))) {
		TRACE("Malloc fails for perm_c[].");
		return FALSE;
	}

    // Initialize the statistics variables
    StatInit(&stat);

	dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);

    if (info == 0) {
		for (i = 0; i < m_rows; i++) {
			x.m_dv[i] = rhs[i];
		}

		Lstore = (SCformat*) L.Store;
		Ustore = (NCformat*) U.Store;
    	TRACE2("No of nonzeros in factor L = %d\n", Lstore->nnz);
    	TRACE2("No of nonzeros in factor U = %d\n", Ustore->nnz);
    	TRACE2("No of nonzeros in L+U = %d\n", Lstore->nnz + Ustore->nnz - m_cols);
    	TRACE2("FILL ratio = %.1f\n", (float)(Lstore->nnz + Ustore->nnz - m_cols) / m_nz);
	
		dQuerySpace(&L, &U, &mem_usage);
		TRACE2("L\\U MB %.3f\ttotal MB needed %.3f\n",
			mem_usage.for_lu/1e6, mem_usage.total_needed/1e6);
    } else {
		TRACE("dgssv() error returns INFO= %d\n", info);
		if (info <= m_cols) { // factorization completes
			dQuerySpace(&L, &U, &mem_usage);
			TRACE("L\\U MB %.3f\ttotal MB needed %.3f\n", mem_usage.for_lu/1e6, mem_usage.total_needed/1e6);
		}
	}

    StatFree(&stat);

    SUPERLU_FREE (rhs);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
    Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);

	if (info != 0) {
		return FALSE;
	}
#endif

	return TRUE;

/*
	int i;
	_INTEGER_t* rowIndex;
	_INTEGER_t* columns;
	_DOUBLE_PRECISION_t* values;
	_MKL_DSS_HANDLE_t handle;
	_INTEGER_t error;
	int nRhs = 1;

    rowIndex = (_INTEGER_t*)MyAlloc((m_rows+1)*sizeof(_INTEGER_t));
    columns = (_INTEGER_t*)MyAlloc(m_nz*sizeof(_INTEGER_t));
    values = (_DOUBLE_PRECISION_t*)MyAlloc(m_nz*sizeof(_DOUBLE_PRECISION_t));
	for (i = 0; i < m_rows+1; i++) {
		rowIndex[i] = m_cr_rowptr[i]+1;
	}
	for (i = 0; i < m_nz; i++) {
		columns[i] = m_cr_colind[i]+1;
		values[i] = m_cr_val[i];
	}
	
	int opt = MKL_DSS_DEFAULTS;
	int sym = MKL_DSS_NON_SYMMETRIC;
	int type = MKL_DSS_POSITIVE_DEFINITE;
	
	// Initialize the solver
	error = dss_create(handle, opt);
	if (error != MKL_DSS_SUCCESS) { AfxMessageBox("dss_create error\r\n"); return; }
	// Define the non-zero structure of the matrix
	error = dss_define_structure(handle, sym, rowIndex, m_rows, m_cols, columns, m_nz);
	if (error != MKL_DSS_SUCCESS) { AfxMessageBox("dss_define_structure error\r\n"); return; }
	// Reorder the matrix
	error = dss_reorder(handle, opt, 0);
	if (error != MKL_DSS_SUCCESS) { AfxMessageBox("dss_reorder error\r\n"); return; }
	// Factor the matrix
	error = dss_factor_real(handle, type, values);
	if (error != MKL_DSS_SUCCESS) { AfxMessageBox("dss_factor_real error\r\n"); return; }

	// Get the solution vector
	error = dss_solve_real(handle, opt, b.m_dv, nRhs, x.m_dv);
	if (error != MKL_DSS_SUCCESS) { AfxMessageBox("dss_solve_real error\r\n"); return; }
	
	// Get the determinant
	//error = dss_statistics(handle, opt, statIn, statOut);
	//if (error != MKL_DSS_SUCCESS) { AfxMessageBox("dss_statistics error\r\n"); return; }
	
	// Deallocate solver storage
	error = dss_delete(handle, opt);
	if (error != MKL_DSS_SUCCESS) { AfxMessageBox("dss_delete error\r\n"); return; }
	
    MyFree(rowIndex);
    MyFree(columns);
    MyFree(values);
*/
}

BOOL CSMatrix::SolveLinearSystem2(CDVector& b1, CDVector& x1, CDVector& b2, CDVector& x2)
{
#ifdef USE_UMFPACK
#if defined(WIN32) && !defined(WIN64)
    void *Symbolic, *Numeric;
	int status;
	int *Ap, *Ai;
	double *Ax;

    Ap = (int*)MyAllocEx((m_cols+1)*sizeof(int), "CSMatrix::SolveLinearSystem Ap");
    Ai = (int*)MyAllocEx(m_nz*sizeof(int), "CSMatrix::SolveLinearSystem Ai");
    Ax = (double*)MyAllocEx(m_nz*sizeof(double), "CSMatrix::SolveLinearSystem Ax");

	status = umfpack_di_triplet_to_col (m_rows, m_cols, m_nz, m_rowind, m_colind, m_val, Ap, Ai, Ax, (int*)NULL);
    if (status < 0) { ReportError("umfpack_di_triplet_to_col", status); return FALSE; }
	status = umfpack_di_symbolic(m_rows, m_cols, Ap, Ai, Ax, &Symbolic, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_symbolic", status); return FALSE; }
    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_numeric", status); return FALSE; }

    umfpack_di_free_symbolic(&Symbolic);
    
	status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x1.m_dv, b1.m_dv, Numeric, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_solve", status); return FALSE; }
    status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x2.m_dv, b2.m_dv, Numeric, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_solve", status); return FALSE; }

    umfpack_di_free_numeric(&Numeric);

	MyFree(Ap);
	MyFree(Ai);
	MyFree(Ax);
#else
    double Info[UMFPACK_INFO], Control[UMFPACK_CONTROL];
    void *Symbolic, *Numeric;
	UF_long status;
	UF_long *Ap, *Ai;
	double *Ax;
	UF_long* rowind;
	UF_long* colind;
	int i;

    Ap = (UF_long*)MyAllocEx((m_cols+1)*sizeof(UF_long), "CSMatrix::SolveLinearSystem Ap");
    Ai = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem Ai");
    Ax = (double*)MyAllocEx(m_nz*sizeof(double), "CSMatrix::SolveLinearSystem Ax");
	rowind = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem rowind");
	colind = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem colind");

	for (i = 0; i < m_nz; i++) {
		rowind[i] = (UF_long)m_rowind[i];
		colind[i] = (UF_long)m_colind[i];
	}

    umfpack_dl_defaults(Control);
	status = umfpack_dl_triplet_to_col (m_rows, m_cols, m_nz, rowind, colind, m_val, Ap, Ai, Ax, (UF_long*)NULL);
    if (status < 0) { ReportError("umfpack_dl_triplet_to_col", status); return FALSE; }
	status = umfpack_dl_symbolic(m_rows, m_cols, Ap, Ai, Ax, &Symbolic, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_symbolic", status); return FALSE; }
    status = umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_numeric", status); return FALSE; }
    
	status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x1.m_dv, b1.m_dv, Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_solve", status); return FALSE; }
	status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x2.m_dv, b2.m_dv, Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_solve", status); return FALSE; }

    umfpack_dl_free_numeric(&Numeric);
    umfpack_dl_free_symbolic(&Symbolic);

	MyFree(Ap);
	MyFree(Ai);
	MyFree(Ax);
	MyFree(rowind);
	MyFree(colind);
#endif
#endif
#ifdef USE_SUPERLU_MT
	SuperMatrix A;
    NCformat *Astore;
    int *perm_r; // row permutations from partial pivoting
    int *perm_c; // column permutation vector
    SuperMatrix L; // factor L
    SCPformat *Lstore;
    SuperMatrix U; // factor U
    NCPformat *Ustore;
    SuperMatrix B;
	int nrhs, info;
	int nprocs; // maximum number of processors to use.
	int panel_size, relax, maxsup;
	int permc_spec;
	double *rhs;
	superlu_memusage_t superlu_memusage;
	int i;

	if (m_cc_colptr == NULL) {
		GenerateCompInfo();
	}

#ifdef _OPENMP
	#pragma omp parallel
	{
		nprocs = omp_get_num_threads();
	}
#else
	nprocs = 1;
#endif
	TRACE2("nprocs = %d\n", nprocs);
	//
	nrhs = 2;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    maxsup = sp_ienv(3);

    dCreate_CompCol_Matrix(&A, m_rows, m_cols, m_nz, m_cc_val, m_cc_rowind, m_cc_colptr, SLU_NC, SLU_D, SLU_GE);
    Astore = (NCformat*)A.Store;
    TRACE2("Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);
    
    if (!(rhs = doubleMalloc(m_rows * nrhs))) {
		TRACE("Malloc fails for rhs[].");
		return FALSE;
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i] = b1.m_dv[i];
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i + m_rows] = b2.m_dv[i];
	}

    dCreate_Dense_Matrix(&B, m_rows, nrhs, rhs, m_rows, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m_rows))) {
		TRACE("Malloc fails for perm_r[].");
		return FALSE;
	}
    if (!(perm_c = intMalloc(m_cols))) {
		TRACE("Malloc fails for perm_c[].");
		return FALSE;
	}

    // Get column permutation vector perm_c[], according to permc_spec:
    //   permc_spec = 0: natural ordering 
    //   permc_spec = 1: minimum degree ordering on structure of A'*A
    //   permc_spec = 2: minimum degree ordering on structure of A'+A
    //   permc_spec = 3: approximate minimum degree for unsymmetric matrices
    permc_spec = 3;
    get_perm_c(permc_spec, &A, perm_c);

	pdgssv(nprocs, &A, perm_c, perm_r, &L, &U, &B, &info);

    if (info == 0) {
		for (i = 0; i < m_rows; i++) {
			x1.m_dv[i] = rhs[i];
		}
		for (i = 0; i < m_rows; i++) {
			x2.m_dv[i] = rhs[i + m_rows];
		}

		Lstore = (SCPformat*)L.Store;
		Ustore = (NCPformat*)U.Store;
    	TRACE2("#NZ in factor L = %d\n", Lstore->nnz);
    	TRACE2("#NZ in factor U = %d\n", Ustore->nnz);
    	TRACE2("#NZ in L+U = %d\n", Lstore->nnz + Ustore->nnz - L.ncol);
	
		superlu_dQuerySpace(nprocs, &L, &U, panel_size, &superlu_memusage);
		TRACE2("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
			superlu_memusage.for_lu/1024/1024, 
			superlu_memusage.total_needed/1024/1024,
			superlu_memusage.expansions);
    }

    SUPERLU_FREE (rhs);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
	Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_SCP(&L);
    Destroy_CompCol_NCP(&U);

	if (info != 0) {
		return FALSE;
	}
#endif
#ifdef USE_SUPERLU
	SuperMatrix A;
    NCformat *Astore;
    int *perm_r; // row permutations from partial pivoting
    int *perm_c; // column permutation vector
    SuperMatrix L; // factor L
    SCformat *Lstore;
    SuperMatrix U; // factor U
    NCformat *Ustore;
    SuperMatrix B;
	int nrhs, info;
	double *rhs;
    mem_usage_t mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;
	int i;

	if (m_cc_colptr == NULL) {
		GenerateCompInfo();
	}

	set_default_options(&options);

	nrhs = 2;

    dCreate_CompCol_Matrix(&A, m_rows, m_cols, m_nz, m_cc_val, m_cc_rowind, m_cc_colptr, SLU_NC, SLU_D, SLU_GE);
    Astore = (NCformat*)A.Store;
    TRACE2("Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);
    
    if (!(rhs = doubleMalloc(m_rows * nrhs))) {
		TRACE("Malloc fails for rhs[].");
		return FALSE;
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i] = b1.m_dv[i];
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i + m_rows] = b2.m_dv[i];
	}

    dCreate_Dense_Matrix(&B, m_rows, nrhs, rhs, m_rows, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m_rows))) {
		TRACE("Malloc fails for perm_r[].");
		return FALSE;
	}
    if (!(perm_c = intMalloc(m_cols))) {
		TRACE("Malloc fails for perm_c[].");
		return FALSE;
	}

    // Initialize the statistics variables
    StatInit(&stat);

	dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);

    if (info == 0) {
		for (i = 0; i < m_rows; i++) {
			x1.m_dv[i] = rhs[i];
		}
		for (i = 0; i < m_rows; i++) {
			x2.m_dv[i] = rhs[i + m_rows];
		}

		Lstore = (SCformat*) L.Store;
		Ustore = (NCformat*) U.Store;
    	TRACE2("No of nonzeros in factor L = %d\n", Lstore->nnz);
    	TRACE2("No of nonzeros in factor U = %d\n", Ustore->nnz);
    	TRACE2("No of nonzeros in L+U = %d\n", Lstore->nnz + Ustore->nnz - m_cols);
    	TRACE2("FILL ratio = %.1f\n", (float)(Lstore->nnz + Ustore->nnz - m_cols) / m_nz);
	
		dQuerySpace(&L, &U, &mem_usage);
		TRACE2("L\\U MB %.3f\ttotal MB needed %.3f\n",
			mem_usage.for_lu/1e6, mem_usage.total_needed/1e6);
    } else {
		TRACE("dgssv() error returns INFO= %d\n", info);
		if (info <= m_cols) { // factorization completes
			dQuerySpace(&L, &U, &mem_usage);
			TRACE("L\\U MB %.3f\ttotal MB needed %.3f\n", mem_usage.for_lu/1e6, mem_usage.total_needed/1e6);
		}
	}

    StatFree(&stat);

    SUPERLU_FREE (rhs);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
	Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);

	if (info != 0) {
		return FALSE;
	}
#endif

	return TRUE;
}

BOOL CSMatrix::SolveLinearSystem3(CDVector& b1, CDVector& x1, CDVector& b2, CDVector& x2, CDVector& b3, CDVector& x3)
{
#ifdef USE_UMFPACK
#if defined(WIN32) && !defined(WIN64)
    void *Symbolic, *Numeric;
	int status;
	int *Ap, *Ai;
	double *Ax;

	/*{
		FILE* fp;
		int i;
		fp = fopen("mtx.txt", "w");
		fprintf(fp, "%d %d %d\n", m_rows, m_cols, m_nz);
		for (i = 0; i < m_nz; i++) {
			fprintf(fp, "%d %d %e\n", m_rowind[i], m_colind[i], m_val[i]);
		}
		fclose(fp);
	}*/

    Ap = (int*)MyAllocEx((m_cols+1)*sizeof(int), "CSMatrix::SolveLinearSystem Ap");
    Ai = (int*)MyAllocEx(m_nz*sizeof(int), "CSMatrix::SolveLinearSystem Ai");
    Ax = (double*)MyAllocEx(m_nz*sizeof(double), "CSMatrix::SolveLinearSystem Ax");

	status = umfpack_di_triplet_to_col (m_rows, m_cols, m_nz, m_rowind, m_colind, m_val, Ap, Ai, Ax, (int*)NULL);
    if (status < 0) { ReportError("umfpack_di_triplet_to_col", status); return FALSE; }
	status = umfpack_di_symbolic(m_rows, m_cols, Ap, Ai, Ax, &Symbolic, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_symbolic", status); return FALSE; }
    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_numeric", status); return FALSE; }

    umfpack_di_free_symbolic(&Symbolic);

    status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x1.m_dv, b1.m_dv, Numeric, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_solve", status); return FALSE; }
    status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x2.m_dv, b2.m_dv, Numeric, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_solve", status); return FALSE; }
    status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x3.m_dv, b3.m_dv, Numeric, (double*)NULL, (double*)NULL);
    if (status < 0) { ReportError("umfpack_di_solve", status); return FALSE; }

    umfpack_di_free_numeric(&Numeric);

	MyFree(Ap);
	MyFree(Ai);
	MyFree(Ax);
#else
    double Info[UMFPACK_INFO], Control[UMFPACK_CONTROL];
    void *Symbolic, *Numeric;
	UF_long status;
	UF_long *Ap, *Ai;
	double *Ax;
	UF_long* rowind;
	UF_long* colind;
	int i;

    Ap = (UF_long*)MyAllocEx((m_cols+1)*sizeof(UF_long), "CSMatrix::SolveLinearSystem Ap");
    Ai = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem Ai");
    Ax = (double*)MyAllocEx(m_nz*sizeof(double), "CSMatrix::SolveLinearSystem Ax");
	rowind = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem rowind");
	colind = (UF_long*)MyAllocEx(m_nz*sizeof(UF_long), "CSMatrix::SolveLinearSystem colind");

	for (i = 0; i < m_nz; i++) {
		rowind[i] = (UF_long)m_rowind[i];
		colind[i] = (UF_long)m_colind[i];
	}

    umfpack_dl_defaults(Control);
	status = umfpack_dl_triplet_to_col (m_rows, m_cols, m_nz, rowind, colind, m_val, Ap, Ai, Ax, (UF_long*)NULL);
    if (status < 0) { ReportError("umfpack_dl_triplet_to_col", status); return FALSE; }
	status = umfpack_dl_symbolic(m_rows, m_cols, Ap, Ai, Ax, &Symbolic, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_symbolic", status); return FALSE; }
    status = umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_numeric", status); return FALSE; }
    
	status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x1.m_dv, b1.m_dv, Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_solve", status); return FALSE; }
	status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x2.m_dv, b2.m_dv, Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_solve", status); return FALSE; }
	status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x3.m_dv, b3.m_dv, Numeric, Control, Info);
    if (status < 0) { ReportError("umfpack_dl_solve", status); return FALSE; }

    umfpack_dl_free_numeric(&Numeric);
    umfpack_dl_free_symbolic(&Symbolic);

	MyFree(Ap);
	MyFree(Ai);
	MyFree(Ax);
	MyFree(rowind);
	MyFree(colind);
#endif
#endif
#ifdef USE_SUPERLU_MT
	SuperMatrix A;
    NCformat *Astore;
    int *perm_r; // row permutations from partial pivoting
    int *perm_c; // column permutation vector
    SuperMatrix L; // factor L
    SCPformat *Lstore;
    SuperMatrix U; // factor U
    NCPformat *Ustore;
    SuperMatrix B;
	int nrhs, info;
	int nprocs; // maximum number of processors to use.
	int panel_size, relax, maxsup;
	int permc_spec;
	double *rhs;
	superlu_memusage_t superlu_memusage;
	int i;

	if (m_cc_colptr == NULL) {
		GenerateCompInfo();
	}

#ifdef _OPENMP
	#pragma omp parallel
	{
		nprocs = omp_get_num_threads();
	}
#else
	nprocs = 1;
#endif
	TRACE2("nprocs = %d\n", nprocs);
	//
	nrhs = 3;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    maxsup = sp_ienv(3);

    dCreate_CompCol_Matrix(&A, m_rows, m_cols, m_nz, m_cc_val, m_cc_rowind, m_cc_colptr, SLU_NC, SLU_D, SLU_GE);
    Astore = (NCformat*)A.Store;
    TRACE2("Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);
    
    if (!(rhs = doubleMalloc(m_rows * nrhs))) {
		TRACE("Malloc fails for rhs[].");
		return FALSE;
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i] = b1.m_dv[i];
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i + m_rows] = b2.m_dv[i];
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i + m_rows*2] = b3.m_dv[i];
	}

    dCreate_Dense_Matrix(&B, m_rows, nrhs, rhs, m_rows, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m_rows))) {
		TRACE("Malloc fails for perm_r[].");
		return FALSE;
	}
    if (!(perm_c = intMalloc(m_cols))) {
		TRACE("Malloc fails for perm_c[].");
		return FALSE;
	}

    // Get column permutation vector perm_c[], according to permc_spec:
    //   permc_spec = 0: natural ordering 
    //   permc_spec = 1: minimum degree ordering on structure of A'*A
    //   permc_spec = 2: minimum degree ordering on structure of A'+A
    //   permc_spec = 3: approximate minimum degree for unsymmetric matrices
    permc_spec = 3;
    get_perm_c(permc_spec, &A, perm_c);

	pdgssv(nprocs, &A, perm_c, perm_r, &L, &U, &B, &info);

    if (info == 0) {
		for (i = 0; i < m_rows; i++) {
			x1.m_dv[i] = rhs[i];
		}
		for (i = 0; i < m_rows; i++) {
			x2.m_dv[i] = rhs[i + m_rows];
		}
		for (i = 0; i < m_rows; i++) {
			x3.m_dv[i] = rhs[i + m_rows*2];
		}

		Lstore = (SCPformat*)L.Store;
		Ustore = (NCPformat*)U.Store;
    	TRACE2("#NZ in factor L = %d\n", Lstore->nnz);
    	TRACE2("#NZ in factor U = %d\n", Ustore->nnz);
    	TRACE2("#NZ in L+U = %d\n", Lstore->nnz + Ustore->nnz - L.ncol);
	
		superlu_dQuerySpace(nprocs, &L, &U, panel_size, &superlu_memusage);
		TRACE2("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
			superlu_memusage.for_lu/1024/1024, 
			superlu_memusage.total_needed/1024/1024,
			superlu_memusage.expansions);
    }

    SUPERLU_FREE (rhs);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
	Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_SCP(&L);
    Destroy_CompCol_NCP(&U);

	if (info != 0) {
		return FALSE;
	}
#endif
#ifdef USE_SUPERLU
	SuperMatrix A;
    NCformat *Astore;
    int *perm_r; // row permutations from partial pivoting
    int *perm_c; // column permutation vector
    SuperMatrix L; // factor L
    SCformat *Lstore;
    SuperMatrix U; // factor U
    NCformat *Ustore;
    SuperMatrix B;
	int nrhs, info;
	double *rhs;
    mem_usage_t mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;
	int i;

	if (m_cc_colptr == NULL) {
		GenerateCompInfo();
	}

	set_default_options(&options);

	nrhs = 3;

    dCreate_CompCol_Matrix(&A, m_rows, m_cols, m_nz, m_cc_val, m_cc_rowind, m_cc_colptr, SLU_NC, SLU_D, SLU_GE);
    Astore = (NCformat*)A.Store;
    TRACE2("Dimension %dx%d; # nonzeros %d\n", A.nrow, A.ncol, Astore->nnz);
    
    if (!(rhs = doubleMalloc(m_rows * nrhs))) {
		TRACE("Malloc fails for rhs[].");
		return FALSE;
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i] = b1.m_dv[i];
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i + m_rows] = b2.m_dv[i];
	}
	for (i = 0; i < m_rows; i++) {
		rhs[i + m_rows*2] = b3.m_dv[i];
	}

    dCreate_Dense_Matrix(&B, m_rows, nrhs, rhs, m_rows, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m_rows))) {
		TRACE("Malloc fails for perm_r[].");
		return FALSE;
	}
    if (!(perm_c = intMalloc(m_cols))) {
		TRACE("Malloc fails for perm_c[].");
		return FALSE;
	}

    // Initialize the statistics variables
    StatInit(&stat);

	dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);

    if (info == 0) {
		for (i = 0; i < m_rows; i++) {
			x1.m_dv[i] = rhs[i];
		}
		for (i = 0; i < m_rows; i++) {
			x2.m_dv[i] = rhs[i + m_rows];
		}
		for (i = 0; i < m_rows; i++) {
			x3.m_dv[i] = rhs[i + m_rows*2];
		}

		Lstore = (SCformat*) L.Store;
		Ustore = (NCformat*) U.Store;
    	TRACE2("No of nonzeros in factor L = %d\n", Lstore->nnz);
    	TRACE2("No of nonzeros in factor U = %d\n", Ustore->nnz);
    	TRACE2("No of nonzeros in L+U = %d\n", Lstore->nnz + Ustore->nnz - m_cols);
    	TRACE2("FILL ratio = %.1f\n", (float)(Lstore->nnz + Ustore->nnz - m_cols) / m_nz);
	
		dQuerySpace(&L, &U, &mem_usage);
		TRACE2("L\\U MB %.3f\ttotal MB needed %.3f\n",
			mem_usage.for_lu/1e6, mem_usage.total_needed/1e6);
    } else {
		TRACE("dgssv() error returns INFO= %d\n", info);
		if (info <= m_cols) { // factorization completes
			dQuerySpace(&L, &U, &mem_usage);
			TRACE("L\\U MB %.3f\ttotal MB needed %.3f\n", mem_usage.for_lu/1e6, mem_usage.total_needed/1e6);
		}
	}

    StatFree(&stat);

    SUPERLU_FREE (rhs);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
	Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);

	if (info != 0) {
		return FALSE;
	}
#endif

	return TRUE;
}

void CSMatrix::SaveMatrix(const char* filename, int mode)
{
	int i, j;
	FILE *fp;
	fp = fopen(filename, "w");
	CDMatrix dm_tmp(m_rows, m_cols, this, FALSE);
	if (mode == 0) {
		for (j = 0; j < m_rows; j++) {
			for (i = 0; i < m_cols; i++) {
				fprintf(fp, "%e ", dm_tmp.m_dm[j][i]);
			}
			fprintf(fp, "\n");
		}
	} else {
		for (j = 0; j < m_rows; j++) {
			for (i = 0; i < m_cols; i++) {
				if (dm_tmp.m_dm[j][i] == 0) {
					fprintf(fp, "0 ");
				} else {
					fprintf(fp, "1 ");
				}
			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

void CSMatrix::SaveLUMatrix(const char* filename_L, const char* filename_U, int mode)
{
	int i, j;
	FILE *fp_L, *fp_U;

	CDMatrix dm_L(m_rows, m_cols);
	CDMatrix dm_U(m_rows, m_cols);

	if (m_LU) {
		for (j = 0; j < m_rows; j++) {
			for (i = 0; i < j; i++) {
				dm_L.m_dm[j][i] = m_LU->m_dm[j][i];
			}
			for (i = j; i < m_cols; i++) {
				dm_U.m_dm[j][i] = m_LU->m_dm[j][i];
			}
		}
	}

	fp_L = fopen(filename_L, "w");
	fp_U = fopen(filename_U, "w");
	if (mode == 0) {
		for (j = 0; j < m_rows; j++) {
			for (i = 0; i < m_cols; i++) {
				fprintf(fp_L, "%e ", dm_L.m_dm[j][i]);
				fprintf(fp_U, "%e ", dm_U.m_dm[j][i]);
			}
			fprintf(fp_L, "\n");
			fprintf(fp_U, "\n");
		}
	} else {
		for (j = 0; j < m_rows; j++) {
			for (i = 0; i < m_cols; i++) {
				if (dm_L.m_dm[j][i] == 0) {
					fprintf(fp_L, "0 ");
				} else {
					fprintf(fp_L, "1 ");
				}
				if (dm_U.m_dm[j][i] == 0) {
					fprintf(fp_U, "0 ");
				} else {
					fprintf(fp_U, "1 ");
				}
			}
			fprintf(fp_L, "\n");
			fprintf(fp_U, "\n");
		}
	}
	fclose(fp_L);
	fclose(fp_U);
}

#endif
