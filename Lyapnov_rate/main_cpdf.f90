!"""
!   Y座標はエラー
!   X座標はリャプノフし数
!   とし確率密度関数P(x,y)を作成する。
!   なお範囲はXmin,Xmax,Ymin,Ymaxで表す。
!"""
module calculate
    implicit none
!============================================
    real(8), parameter :: Xmin = -0.2d0
    real(8), parameter :: Xmax = 0.1d0
    real(8), parameter :: Ymin = 0.0d0
    real(8), parameter :: Ymax = 1.0d0
    integer, parameter :: dot_max =4950
    integer, parameter :: dot_num = 1
    real(8), parameter :: dx = 0.25d-2
    real(8), parameter :: dy = 0.5d-2
!    real(8), parameter :: dx = 1.d-2
!    real(8), parameter :: dy = 1.d-2
!============================================
    integer, parameter :: NXmin = -80
    integer, parameter :: NXmax = 40
    integer, parameter :: NYmin = 0
    integer, parameter :: NYmax = 200
!    integer, parameter :: NXmin = -360
!    integer, parameter :: NXmax = 40
!    integer, parameter :: NYmin = 0
!    integer, parameter :: NYmax = 200
    integer ix,iy
    real(8) idx,idy
    
  contains
    subroutine calu_probability_density_function
    
    end subroutine calu_probability_density_function
    subroutine output(Pxy_f,Px_f,istep)
    !----------------------------------------------------------------------------
    !■ 出力
    !----------------------------------------------------------------------------
        real(kind=8) :: Pxy_f(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Px_f(NXmin:NXmax)
        integer istep
        character(6) :: cistep
        
        write(cistep,'(i4.4)') istep
        open(22,file='./pdf_renban/probability_density_function.'//cistep)
        open(23,file='./c-pdf_renban/conditiona_pdf.'//cistep)
        do iX=NXmin,NXmax
        do iY=NYmin,NYmax
            write(22,'(3e14.6)') dx*dble(iX),dy*dble(iY),Pxy_f(iX,iY)
        enddo
        write(22,*)
        enddo
        do iX=NXmin,NXmax
            write(23,'(2e14.6)') dx*dble(iX),Px_f(iX)
        enddo
        close(22)
        close(23)
    end subroutine output
    
end module calculate

program main
    use calculate
    implicit none
    
    real(8) P_xy(NXmin:NXmax, NYmin:NYmax)
    real(8)  P_x(NXmin:NXmax)
!    real(8)  P_y(-100:0)
    real(8) lyap(dot_max,2)
    real(8)  err(dot_max,2)
    real(8) dot_num_range
    integer tmp_x,tmp_y
    integer size0
    integer istep
    character(6) :: cistep
    integer i,j,k
    
    
    do istep=0,1500
        write(*,*) istep
        write(cistep,'(i4.4)') istep
        open(21,file='./data_renban3/lyap_epoch_from_python.'//cistep)
        do i=1,dot_max
            read(21,*) lyap(i,2),err(i,1),err(i,2),lyap(i,1)
        enddo
        close(21)
        P_xy(:,:) = 0.d0
        P_x(:) = 0.d0
        do i=1,dot_max
           if( Xmin <lyap(i,2) .and. lyap(i,2) < Xmax ) then
           if( Ymin < err(i,2) .and. err(i,2) < Ymax ) then
                !tmp_x = ceiling(lyap(i,2)/dx)
                !tmp_y = ceiling( err(i,2)/dy)
                tmp_x = nint(lyap(i,2)/dx)
                tmp_y = nint( err(i,2)/dy)
                !if(tmp_x<=0) tmp_x=tmp_x -1
                !if(tmp_y<=0) tmp_y=tmp_y -1
                P_xy(tmp_x,tmp_y) = P_xy(tmp_x,tmp_y) + 1.d0
            endif
            endif
        enddo
        
        do iX=NXmin,NXmax
            do iY=NYmin,NYmax
                P_x(ix) = P_x(ix) + P_xy(ix,iy)
            enddo
        enddo
!        enddo
!        enddo
!        write(*,*) P_xy
        dot_num_range=0.d0
!        do i=NYmin,NYmax
!        do j=NXmin,NXmax
!            dot_num_range= dot_num_range+P(j,i)
!        enddo
!        enddo
!        P_xy=P_xy/dble(dot_num_range)
        P_xy=P_xy/dble(dot_num)
        P_x=P_x/dble(dot_num)
        call output(P_xy,P_x,istep)
    enddo
end program