program main
    implicit none
    real(8) a0,a1,a2
    integer b0,b1,b2
    
    
    a0=0.002345
    a1=-1.d0*a0
    
    b0=ceiling(a0/0.01d0)
    b1=ceiling(a1/0.01d0)
    
    write(*,*) "b0",b0
    write(*,*) "b1",b1
end program main