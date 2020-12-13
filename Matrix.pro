CONFIG -= qt

TEMPLATE = lib
DEFINES += MATRIX_LIBRARY

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


HEADERS += \
    Matrix_global.h \
    Matrix.hpp \
    CUDA_src/CUDA_global.h \
    CUDA_src/CUDA_setup.h \
    CUDA_src/CUDA_matrix_operators.h

# CUDA settings
CUDA_SOURCES += CUDA_src/CUDA_global.cu\
                CUDA_src/CUDA_setup.cu \
                CUDA_src/CUDA_matrix_operators.cu \

CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1"
SYSTEM_NAME = x64              #NB: SYSTEM DEPENDENT
SYSTEM_TYPE = 64               #SAME HERE
CUDA_ARCH = sm_61              #Compute capability of the GPU (here GTX Geforce 1050 - Compute capability 6.1 so sm_61)
NVCC_OPTIONS = --use_fast_math #Cuda compiler options

#Enable CUDA headers
INCLUDEPATH += $$CUDA_DIR/include

#CUDA librairies headers
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME
#Required libraires added here
LIBS += -lcuda -lcudart

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


# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target
