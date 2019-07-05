!"""
!   Y座標はエラー
!   X座標はリャプノフし数
!   とし確率密度関数P(x,y)を作成する。
!   なお範囲はXmin,Xmax,Ymin,Ymaxで表す。
!"""
module calculate
    implicit none
!============================================
    real(8), parameter :: Xmin = -0.9d0
    real(8), parameter :: Xmax = 0.1d0
    real(8), parameter :: Ymin = 0.d0
    real(8), parameter :: Ymax = 1.0d0
    integer, parameter :: dot_max =4800
    integer, parameter :: dot_num = 4800
    real(8), parameter :: dx = 0.25d-2
    real(8), parameter :: dy = 0.5d-2
!============================================
    
    integer, parameter :: NXmin = -360
    integer, parameter :: NXmax = 40
    integer, parameter :: NYmin = 0
    integer, parameter :: NYmax = 200
    integer ix,iy
    real(8) idx,idy
    
  contains
    subroutine calu_probability_density_function
    
    end subroutine calu_probability_density_function
    subroutine output(P,istep)
    !----------------------------------------------------------------------------
    !■ 出力
    !----------------------------------------------------------------------------
        real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
        integer istep
        character(6) :: cistep
        
        write(cistep,'(i4.4)') istep
        open(22,file='./data_renban2/probability_density_function.'//cistep)
        do iX=NXmin,NXmax
        do iY=NYmin,NYmax
            write(22,'(3e14.6)') dx*dble(iX),dy*dble(iY),P(iX,iY)
        enddo
        write(22,*)
        enddo
        close(22)
    end subroutine output
    
end module calculate

program main
    use calculate
    implicit none
    
    real(8) P_xy(NXmin:NXmax, NYmin:NYmax)
    real(8)  P_x(-100:0)
    real(8)  P_y(-100:0)
    real(8) lyap(dot_max,2)
    real(8)  err(dot_max,2)
    real(8) dot_num_range
    integer tmp_x,tmp_y
    integer size0
    integer istep
    character(6) :: cistep
    integer i,j,k
    
    
    do istep=1,1500
        write(*,*) istep
        write(cistep,'(i4.4)') istep
        open(21,file='./data_renban3/lyap_epoch_from_python.'//cistep)
 !       open(21, file='1.dat')
        inquire( 21, pos=size0 )
!        write(*,*) size0
        do i=1,dot_max
            read(21,*) lyap(i,2),err(i,1),err(i,2),lyap(i,1)
        enddo
        close(21)
!        write(*,*) lyap
        
        P_xy(:,:) = 0
!        do ix = NXmin,NXmax
!        do iy = NYmin,NYmax
        do i=1,dot_max
!            write(*,*) lyap(i,2),err(i,2)
           if( Xmin <lyap(i,2) .and. lyap(i,2) < Xmax ) then
           if( Ymin < err(i,2) .and. err(i,2) < Ymax ) then
                tmp_x = ceiling(lyap(i,2)/dx)
                tmp_y = ceiling( err(i,2)/dy)
!                write(*,*) tmp_x,tmp_y
                if(tmp_x<0) tmp_x=tmp_x -1
                
                if(tmp_y<0) tmp_y=tmp_y -1
            
                P_xy(tmp_x,tmp_y) = P_xy(tmp_x,tmp_y) + 1.d0
                
            endif
            endif
            
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
        call output(P_xy,istep)
    enddo
end program