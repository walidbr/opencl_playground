all:
	gcc -o vector_multiply vector_multiply.c -lOpenCL
	gcc -o matrix_multiply matrix_multiply.c -lOpenCL
clean: 
	rm -rf vector_multiply matrix_multiply 
