# Makefile to build SPINS, generalized for multiple systemd

# The "system.mk" file in the current directory will contain
# system-specific make variables, most notably the
# C/C++ compiler/linker, 
include system.mk


# Compile with debugging flags
DEBUG?=false
# Compile with optimizations
OPTIM?=true
# Compile with extra optimizations that may require significantly
# longer for compilation
SLOW_OPTIM?=false

# If MPICXX isn't separately defined, then set it to be the same
# as CXX
ifeq ($(strip $(MPICXX)),)
   MPICXX:=$(CXX)
endif

# Assemble the CFLAGS
CFLAGS:=$(SYSTEM_CFLAGS) $(MPI_CFLAGS) $(SYSTEM_CXXFLAGS)
LDFLAGS:=$(SYSTEM_LDFLAGS)
ifeq ($(DEBUG),true)
   CFLAGS:=$(CFLAGS) $(DEBUG_CFLAGS)
   LDFLAGS:=$(LDFLAGS) $(DEBUG_LDFLAGS)
endif
ifeq ($(OPTIM),true)
   CFLAGS:=$(CFLAGS) $(OPTIM_CFLAGS)
   LDFLAGS:=$(LDFLAGS) $(OPTIM_LDFLAGS)
   ifeq ($(SLOW_OPTIM),true)
      CFLAGS:=$(CFLAGS) $(EXTRA_OPTIM_CFLAGS)
      LDFLAGS:=$(LDFLAGS) $(EXTRA_OPTIM_LDFLAGS)
   endif
endif

INCLUDE_DIRS := -I../include $(MPI_INCDIR) $(LAPACK_INCDIR) $(BLITZ_INCDIR) $(FFTW_INCDIR) $(UMF_INCDIR)
CFLAGS := $(CFLAGS) $(INCLUDE_DIRS)

LIB_DIRS := -L../lib $(MPI_LIBDIR) $(LAPACK_LIBDIR) $(BLITZ_LIBDIR) $(FFTW_LIBDIR) $(UMF_LIBDIR)
LDLIBS := -lfftw3 $(MPI_LIB) -lumfpack -lamd -lboost_program_options $(LAPACK_LIB) -lblitz -lm -lstdc++ 
LDFLAGS := $(LDFLAGS) $(LIB_DIRS) 

.PHONY: all 
all: tests/test_deriv_x tests/test_write_x tests/test_esolve_x tests/test_heat_x tests/test_ode_x tests/test_ns_x

.PHONY: clean
clean:
	rm -f *.o tests/*.o cases/*.o

objfiles: $(shell ls *.cpp | sed -e 's/cpp/o/g')

NSIntegrator.o: NSIntegrator.cpp NSIntegrator_impl.cc

tests/test%.o: tests/tests%.cpp
	$(MPICXX) $(CFLAGS) -o $@ -c $<

tests/test%_x: tests/test%.o TArray.o Parformer.o T_util.o  ESolver.o Timestep.o NSIntegrator.o BaseCase.o Science.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o Options.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS) 

cases/%.o: cases/%.cpp NSIntegrator_impl.cc NSIntegrator.hpp
	$(MPICXX) $(CFLAGS) -o $@ -c  $<


nonhydro_x: nonhydro_sw.o TArray.o T_util.o Parformer.o Splits.o Par_util.o Split_reader.o 
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

derek_x: derek.o TArray.o T_util.o Parformer.o ESolver.o Timestep.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o 
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

NHchannel_x: NHchannel.o TArray.o Parformer.o T_util.o gmres.o gmres_2d_solver.o gmres_1d_solver.o Splits.o Par_util.o gmres.o grad.o multigrid.o chebmat.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

NHchannel_2lr_x: NHchannel_2lr.o TArray.o Parformer.o T_util.o gmres.o gmres_2d_solver.o gmres_1d_solver.o Splits.o Par_util.o gmres.o grad.o multigrid.o chebmat.o Timestep.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

cases/%.x: cases/%.o TArray.o T_util.o Parformer.o ESolver.o Timestep.o NSIntegrator.o BaseCase.o Science.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o Options.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

cases/%_x: cases/%.o TArray.o T_util.o Parformer.o ESolver.o Timestep.o NSIntegrator.o BaseCase.o Science.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o Options.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o : %.cpp	*.hpp
	$(MPICXX) $(CFLAGS) -o $@ -c $< 

