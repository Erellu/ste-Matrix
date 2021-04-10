
#######################################################################################################################
# This qmake file is adapted from this StackOverflow answer by Yellow (https://stackoverflow.com/users/1511627/yellow)#
# https://stackoverflow.com/questions/12266264/compiling-cuda-code-in-qt-creator-on-windows                           #
#######################################################################################################################


# Qt config
TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += qt

QT += core
QT -= gui

#######################################################

# Enables CUDA functions in ste::Matrix
DEFINES += STE_MATRIX_ALLOW_GPU


#######################################################

# Avoid conflicts between CUDA and MSVC
QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:msvcrtd.lib
QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:libcmt.lib



## Avoid conflicts between CUDA and MSVC
QMAKE_CFLAGS_DEBUG      += /MTd
QMAKE_CFLAGS_RELEASE    += /MT
QMAKE_CXXFLAGS_DEBUG    += /MTd
QMAKE_CXXFLAGS_RELEASE  += /MT

#######################################################
#                       C++ 

# SOURCES += #Add your source files here

HEADERS += ../ste-Matrix/include/Matrix.hpp \ #Path to the ste::Matrix header file
           #Add your headers here


#######################################################
#                       CUDA


#CUDA headers
HEADERS += ../ste-Matrix/include/CUDA_src/CUDA_global.h \           #Path to the ste::Matrix CUDA header files
           ../ste-Matrix/include/CUDA_src/CUDA_matrix_operators.h \

# CUDA source files
CUDA_SOURCES +=  ../ste-Matrix/include/CUDA_src/CUDA_global.cu \            #Path to the ste::Matrix CUDA source files
                 ../ste-Matrix/include/CUDA_src/CUDA_matrix_operators.cu \

OTHER_FILES += CUDA_SOURCES



CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1" #Path to your CUDA installation. Do not forget to check the version.
SYSTEM_NAME = x64              #NB: SYSTEM DEPENDENT
SYSTEM_TYPE = 64               #SAME HERE
CUDA_ARCH = sm_61              #Compute capability of the GPU (here GTX Geforce 1050 - Compute capability 6.1 so sm_61)
NVCC_OPTIONS = --use_fast_math #Cuda compiler options

#Enable CUDA headers
INCLUDEPATH += $$CUDA_DIR/include

#CUDA librairies headers
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME

#Required libraires added here. ste::Matrix requires these 3 CUDA ones.
LIBS += -lcuda -lcudart -lcublas

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

#Configuration of the CUDA compiler
CONFIG(debug, debug|release) {
       # Debug mode
       cuda_d.input = CUDA_SOURCES
       cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
       cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
       cuda_d.dependency_type = TYPE_C
       QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
       # Release mode
       cuda.input = CUDA_SOURCES
       cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
       cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
       cuda.dependency_type = TYPE_C
       QMAKE_EXTRA_COMPILERS += cuda
}

