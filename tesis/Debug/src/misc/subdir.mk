################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/misc/functions.cpp 

OBJS += \
./src/misc/functions.o 

CPP_DEPS += \
./src/misc/functions.d 


# Each subdirectory must supply rules for building sources it contributes
src/misc/%.o: ../src/misc/%.cpp src/misc/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


