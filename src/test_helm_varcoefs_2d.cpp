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

#define Nx 256
#define Ny 128

// Grid lengths

#define Lx 2000.0
#define Ly 4000.0  //Need to look like 'double' otherwise multigrid freaks out.

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

   x = (ii+0.5)/Nx*2*M_PI;
   y = -cos(M_PI*ii/(Ny-1)); //Ask Chris: why do we put ii and not kk here?

   //adjust from standard grids to physical grids
   x = (Lx/2/M_PI)*x(ii);
   y = Ly*((y(ii)+1)/2);
 

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

   // Allocate space for variable coefficients.
   DTArray cx(local_lbound,local_extent,local_storage),
           cxx(local_lbound,local_extent,local_storage),
           cy(local_lbound,local_extent,local_storage),
           cyy(local_lbound,local_extent,local_storage);

   // set variable coefficients, based on a diffusivity of
   // a(x,y) = cos(pi*y)*sin(3*x) //Ack! this vanishes in the domain!
  // cx = 0*3*cos(M_PI*y(kk))*cos(3*x(ii)); // coef of Dx
  // cxx = 2+0*cos(M_PI*y(kk))*sin(3*x(ii));  //coef of Dxx
   //cxx = 3+cos(M_PI*y(kk)); // works.
     cxx = 3+cos(M_PI*y(kk)/Ly)*sin(3*2*M_PI*x(ii)/Lx);
     cx = ((3*2*M_PI)/Lx)*cos(M_PI*y(kk)/Ly)*cos(3*2*M_PI*x(ii)/Lx);
     
  // cy = 0*(-M_PI*sin(M_PI*y(kk))*sin(3*x(ii)));   //coef of Dy
 //  cyy = 2+0*cos(M_PI*y(kk))*sin(3*x(ii));  //coef of Dyy
   //cyy = 3+cos(M_PI*y(kk)); //works.
     cyy = 3+cos(M_PI*y(kk)/Ly)*sin(3*2*M_PI*x(ii)/Lx);
     cy = (-M_PI/Ly)*sin(M_PI*y(kk)/Ly)*sin(3*2*M_PI*x(ii)/Lx);

   // Write grid to disk   
   u = x(ii) + 0*kk;      //Need 0*kk so blitz doesn't interpret as 1d vector
   write_array(u,"xgrid");
   write_reader(u,"xgrid");
 
   u = y(kk);
   write_array(u,"ygrid");
   write_reader(u,"ygrid");
  
   // set exact solution to Poisson problem with homogenous bc's in y.
   u_exact = cos(M_PI*y(kk))*sin(x(ii));

   write_array(u_exact,"u_exact");
   write_reader(u_exact,"u_exact");

   // set right-hand side as Helmholtzian of exact solution (check this)
   //rhs = 3*cos(M_PI*y(kk))*cos(M_PI*y(kk))*cos(3*x(ii)) + M_PI*M_PI*sin(M_PI*y(kk))*sin(M_PI*y(kk))*sin(3*x(ii))*sin(x(ii)) - cos(M_PI*y(kk))*cos(M_PI*y(kk))*sin(3*x(ii))*sin(x(ii)) - M_PI*M_PI*cos(M_PI*y(kk))*cos(M_PI*y(kk))*sin(3*x(ii))*sin(x(ii)) - cos(M_PI*y(kk))*sin(x(ii)) ;

  // rhs = -cos(M_PI*y(kk))*cos(M_PI*y(kk))*sin(3*x(ii))*sin(x(ii)) - cos(M_PI*y(kk))*cos(M_PI*y(kk))*sin(3*x(ii))*M_PI*M_PI*sin(x(ii)) - cos(M_PI*y(kk))*sin(x(ii));

  // rhs = cos(M_PI*y(kk))*sin(x(ii));

   rhs = 1+cos(2*M_PI*y(kk)/Ly)*sin(2*2*M_PI*x(ii)/Lx);

   write_array(rhs,"rhs");
   write_reader(rhs,"rhs");

   // Build gradient operator, calling seq: szx,szy,szz, s_exp in x,y,z
   Grad mygrad(Nx,1,Ny,FOURIER,FOURIER,CHEBY);
    
   mygrad.set_jac(firstDim,firstDim,2*M_PI/Lx);
   mygrad.set_jac(secondDim,secondDim,1.0);
   mygrad.set_jac(thirdDim,thirdDim,-2/Ly); //Cheby grids need the - 
   
   // create gmres-2d solver interface
   Cheb_2dmg a(Nx,Ny);
   
   double dbc=0.0;
   double nbc=1.0;

   // set variable coefs (must do before set_grad)
   a.set_ci(&cx,1);
   a.set_ci(&cxx,2);
   a.set_ci(&cy,3);
   a.set_ci(&cyy,4);

   a.set_grad(&mygrad); // need to do this before set_bc 

   // set_bc, calling sequence is helm, dir., neu., S_EXP ('x' series expans)
   a.set_bc(-1.0,dbc,nbc,FOURIER); // Sign of 'helm' opposite that of 1D code. 

   // Initialize structures that are used by gmres object
   fbox * init_r = a.alloc_resid();
   ubox * final_u = a.alloc_basis();
 
   *init_r->gridbox = rhs;

   // set BC's:
   // Note, can only use blitz indices on RHS, on LHS use range stuff.
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
   int itercount = mysolver.Solve(final_u,init_r,1e-7,100,1);

   write_array(*final_u->gridbox,"output_2d");
   write_reader(*final_u->gridbox,"output_2d",false);

   // free memory of residual and basis
   a.free_resid(init_r);
   a.free_basis(final_u);

   cout << itercount << endl;
    
   MPI_Finalize();
   return 0;
}


