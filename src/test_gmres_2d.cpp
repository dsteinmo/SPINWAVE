#include "TArray.hpp"
#include <blitz/array.h>
#include "T_util.hpp"
#include "Par_util.hpp"
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "Splits.hpp"
#include <random/normal.h>
#include <fstream>
#include "Parformer.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "gmres.hpp"
#include "gmres_2d_solver.hpp"
#include "gmres_2d_solver_impl.hpp"
#include "grad.hpp"

using namespace TArrayn;
using namespace Transformer;

using blitz::Array;
using blitz::TinyVector;
using blitz::GeneralArrayStorage;
using blitz::Range;

using ranlib::Normal;

// Grid size

#define Nx 16
#define Ny 16

// Grid lengths

#define Lx 2*M_PI
#define Ly 1

using namespace std;

// Blitz index placeholders

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

int main(int argc, char ** argv) {
   // Initialize MPI
   MPI_Init(&argc, &argv);

   // make grids
   Array<double,1> x(Nx), y(Ny);

   x = (ii+0.5)/Nx*Lx;
   y = -cos(M_PI*ii/(Ny-1));

   // Get parameters for local array storage

   TinyVector<int,3> local_lbound, local_extent;
   GeneralArrayStorage<3> local_storage;

   local_lbound = alloc_lbound(Nx,1,Ny);
   local_extent = alloc_extent(Nx,1,Ny);
   local_storage = alloc_storage(Nx,1,Ny);

   // Allocate space for solution, u, rhs, and exact soln
   DTArray u(local_lbound,local_extent,local_storage),
           rhs(local_lbound,local_extent,local_storage),
           u_exact(local_lbound,local_extent,local_storage);

   // Write grid to disk   
   u = x(ii) + 0*kk;      //Need 0*kk so blitz doesn't interpret as 1d vector
   write_array(u,"xgrid");
   write_reader(u,"xgrid");
 
   u = y(kk);
   write_array(u,"ygrid");
   write_reader(u,"ygrid");
  
   // set exact solution to Poisson problem with homogenous bc's in y.
   u_exact = sin(M_PI*y(kk))*cos(x(ii));

   write_array(u_exact,"u_exact");
   write_reader(u_exact,"u_exact");

   // set right-hand side as Laplacian of exact solution
   rhs = -sin(M_PI*y(kk))*cos(x(ii)) - M_PI*M_PI*sin(M_PI*y(kk))*cos(x(ii));

   write_array(rhs,"rhs");
   write_reader(rhs,"rhs");

   // Build gradient operator, calling seq: szx,szy,szz, s_exp in x,y,z
   Grad mygrad(Nx,1,Ny,FOURIER,FOURIER,CHEBY);
    
   mygrad.set_jac(firstDim,firstDim,1.0);
   mygrad.set_jac(secondDim,secondDim,1.0);
   mygrad.set_jac(thirdDim,thirdDim,1.0);
   
   // create gmres-2d solver interface
   Cheb_2dmg a(Nx,Ny);
   
   double dbc=1.0;
   double nbc=0.0;

   a.set_grad(&mygrad); // need to do this before set_bc 

   // set_bc, calling sequence is helm, dir., neu., S_EXP ('x' series expans)
   a.set_bc(0.0,dbc,nbc,FOURIER); 

   // Initialize structures that are used by gmres object
   fbox * init_r = a.alloc_resid();
   ubox * final_u = a.alloc_basis();
 
   *init_r->gridbox = rhs;

   // Don't need to set boundary conditions in this simple case.
   // but this is where you would do it, e.g.,
   // along y=-1
   (*init_r->gridbox)(Range::all(),
	 Range(0,0),Range(0,0)) = 0;
   // along y=+1
   (*init_r->gridbox)(Range::all(),
	 Range(0,0),Range(Ny-1,Ny-1)) = 0;
   //(I think)

   //Build the solver object with template class
   GMRES_Solver<Cheb_2dmg> mysolver(&a);

   //Calling sequence is lhs,rhs,tol,inner iterations, outer iterations
   int itercount = mysolver.Solve(final_u,init_r,1e-8,10,10);

   write_array(*final_u->gridbox,"output_2d");
   write_reader(*final_u->gridbox,"output_2d",false);

   // free memory of residual and basis
   a.free_resid(init_r);
   a.free_basis(final_u);
 
   MPI_Finalize();
   return 0;
}


