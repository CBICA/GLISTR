///////////////////////////////////////////////////////////////////////////////////////
// DVector.h
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

#ifndef DVECTOR_H
#define DVECTOR_H


#include "SMatrix.h"
#include "DMatrix.h"


class CSMatrix;
class CDMatrix;


class CDVector  
{
public:
	CDVector();
	CDVector(int rows);
	virtual ~CDVector();

	void Init(int rows);
	void Free();
	void Clear();

	void Set(CDVector& X);
	void SetMinus(CDVector& X);

	void Set_APb(double alpha, CSMatrix& A, CDVector& b);
	void Set_APb(double alpha, CDMatrix& A, CDVector& b);
	void Add_APb(double alpha, CSMatrix& A, CDVector& b);
	void Add(CDVector& X);
	void Mult(double s);
	void Set_a_b(double alpha, CDVector& a, double beta, CDVector& b);

	double Get_ata();
	double Get_atb(CDVector& b);
	double Get_atAPb(double alpha, CSMatrix& A, CDVector& b);

	BOOL SaveVector(const char* filename, int mode = 0);
	BOOL LoadVector(const char* filename);

public:
	double* m_dv;
	int m_rows;
};


#endif
