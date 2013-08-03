#include <mpi.h>
#include "TArray.hpp"
#include "T_util.hpp"
#include "gmres.hpp"
#include "gmres_1d_solver.hpp"
#include "gmres_1d_solver_impl.hpp"
#include "Parformer.hpp"
#include <blitz/array.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
/* Tests GMRES for a single-dimension */

using namespace std;
using TArrayn::DTArray;
using Transformer::Trans1D;
using TArrayn::deriv_cheb;
using TArrayn::firstDim; using TArrayn::secondDim; using TArrayn::thirdDim;

extern "C" {
   extern void dgbsv_(int *, int *, int *, int *, double *, 
               int *, int *, double *, int *, int *);
}

blitz::thirdIndex kk;
blitz::secondIndex jj;
blitz::firstIndex ii;
int main(int argc, char ** argv) {
   MPI_Init(&argc,&argv);
   int N = 0;
   if (argc > 1) N = atoi(argv[1]);
   cout << "N = " << N << endl;
   if (N <= 0) N = 32;
   Array<double,1> x(N);
   x = -cos(M_PI*ii/(N-1));
  
  double dbc=1.0;
  double nbc=0.0;
    
  cheb_d2 a(N,2.0); // create gmres-1d solver
  a.set_bc(0.0,dbc,dbc,nbc,nbc); // set boundary conditions and helmholtz
  //cheb_d2 a(N,dbc,dbc,nbc,nbc); // Create the operator-kernel
  printf("Testing bc pair (%.2f, %.2f)...\n",dbc,nbc);
  gridline * init_r = a.alloc_resid();
  gridline * final_z = a.alloc_basis();
  /* Create the true solution, sin(x) */
  Array<double,1> true_soln(N);
  true_soln = sin(M_PI*x(ii));
  cout << "x = " << x << endl
  << "true_soln = " << true_soln << endl;
  /* The second derivative of sin(x) is -sin(x) */
  *init_r->zline = -(M_PI*M_PI)*sin(M_PI*x(kk));

  //(*init_r->zline)(0,0,0) = -(M_PI*MPI)*cos(M_PI*x(ii));
  /* BC's -- general Robin formulation.  The normal is an outward-facing
         normal, so the -1BC gets a -b */
  // Set boundary conditions based on what we know if the true solution
  // From a few lines above, we're using sin(pi*x), so that's what gets
  // used for the BC setting

  // Boundary #1: x(0) = -1
 (*init_r->zline)(0,0,0) = dbc*sin(M_PI*(-1.0)) + // Dirichlet
                           -nbc*M_PI*cos(M_PI*(-1.0)); // Neumann part
  // Boundary #2: x(end) = 1
 (*init_r->zline)(0,0,N-1) = dbc*sin(M_PI*(1.0)) + // Dirichlet
                           nbc*M_PI*cos(M_PI*(1.0)); // Neumann part
 // There is a positive term on the neumann BC for the right boundary
 // because it's an outward derivative

  GMRES_Solver<cheb_d2> solver(&a);
  cout << "init_r = " << *(init_r->zline) << endl;
  //Calling sequence is lhs,rhs,tol,inner ierations, outer iterations
  int itercount = solver.Solve(final_z,init_r,1e-8,10,10);
  /* Now, hopefully all goes well.  */
  printf("Solved in %d iterations. Maximum error %g\n",
        itercount, max(abs((*final_z->zline)(ii,jj,kk)-true_soln(kk))));
 
  Array<double,1> gmres_soln(N);
//  gmres_soln = (*final_z->zline)(ii,jj,kk);

 cout << *(final_z->zline) << endl;
 // Don't free the memory until after it's been printed 
 a.free_resid(init_r);
 a.free_basis(final_z);

 MPI_Finalize();

}
