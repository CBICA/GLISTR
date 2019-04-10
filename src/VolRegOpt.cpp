///////////////////////////////////////////////////////////////////////////////////////
// VolRegOpt.cpp
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

#include "Volume.h"
#include "VolumeBP.h"

#include "VolRegOpt.h"

#include "FFD_table.h"


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
#define DC_REAL float

void get_ninv(DC_REAL*** ninv, int mesh_ex, int mesh_ey, int mesh_ez) {
	int i, j, k, l, m, n;
	int ninv_i, ninv_j, ninv_k;
	DC_REAL u, v, w;
	DC_REAL norm = 1.0 / (mesh_ex*mesh_ey*mesh_ez);
	//
	ninv_k = 0;
	for (n = 0; n < 4; n++) {
		for (k = 0; k < mesh_ez; k++) {
			w = (DC_REAL)k / mesh_ez;
			ninv_j = 0;
			for (m = 0; m < 4; m++) {
				for (j = 0; j < mesh_ey; j++) {
					v = (DC_REAL)j / mesh_ey;
					ninv_i = 0;
					for (l = 0; l < 4; l++) {
						for (i = 0; i < mesh_ex; i++) {
							u = (DC_REAL)i / mesh_ex;
							//ninv[ninv_k][ninv_j][ninv_i] = 1.0 / (FFD_B(3-l, u) * FFD_B(3-m, v) * FFD_B(3-n, w));
							ninv[ninv_k][ninv_j][ninv_i] = norm * FFD_B(3-l, u) * FFD_B(3-m, v) * FFD_B(3-n, w);
							ninv_i++;
						}
					}
					ninv_j++;
				}
			}
			ninv_k++;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


BOOL ComputeFlow3D_TRWS_Decomposed(RVolume& dcv, int wsize, REALV label_sx, REALV label_sy, REALV label_sz, int lmode, REALV alpha_O1, REALV d_O1, REALV alpha_O2, REALV d_O2, REALV gamma, int nIterations,
	RVolume* xx, RVolume* yy, RVolume* zz, RVolume& vx, RVolume& vy, RVolume& vz, int vd_x, int vd_y, int vd_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV in_scv_w_O1F2, REALV in_scv_w_O2F2, REALV in_scv_w_O2F3, double* pEnergy, double* pLowerBound, RVolume* sc, int ffd)
{
	VolumeBP* vbp;

	TRACE2("ComputeFlow3D_TRWS_Decomposed\n");

	vbp = new VolumeBP();
	
	TRACE2("init\n");
	vbp->init(&dcv, wsize, label_sx, label_sy, label_sz, lmode, alpha_O1, d_O1, alpha_O2, d_O2, xx, yy, zz, mesh_ex, mesh_ey, mesh_ez, in_scv_w_O1F2, in_scv_w_O2F2, in_scv_w_O2F3, sc);
	
	TRACE2("ComputeRangeTerm\n");
	vbp->ComputeRangeTerm(gamma);
	
	TRACE2("MessagePassing\n");
	vbp->MessagePassing(1, nIterations, 2, pEnergy, pLowerBound);
	
	TRACE2("ComputeVelocity %d\n", ffd);
	if (ffd == 0) {
		vbp->ComputeVelocity(&vx, &vy, &vz, vd_x, vd_y, vd_z);
	} else {
		vbp->ComputeVelocityFFD(&vx, &vy, &vz, vd_x, vd_y, vd_z);
	}
	
	delete vbp;

	return TRUE;
}

BOOL ComputeFlow3D_FastPD(RVolume& dcv, int wsize, REALV label_s, int lmode, REALV alpha_O1, REALV d_O1,
	RVolume& xx, RVolume& yy, RVolume& zz, RVolume& vx, RVolume& vy, RVolume& vz, int mesh_ex, int mesh_ey, int mesh_ez,
	double* pEnergyPrev, double* pEnergy)
{
	return FALSE;
}
