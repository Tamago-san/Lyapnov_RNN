!-----------------------------------------------------------------------------
!ローレンツモデルのリアプノフ指数を求める。
!
!モデルのタイムスケールの0.01で進む。
!gfortran RNN_LY.f90 rnn_tanh.f90 -o RNN_LY.out -llapack -lblas
!-----------------------------------------------------------------------------
module lyapnov
    implicit none
    integer, parameter :: rc_node = 50
    integer, parameter :: par_node = 3
!    real(8), parameter :: par_a = 16.d0
!    real(8), parameter :: par_b = 40.d0
!    real(8), parameter :: par_c = 4.d0
    real(8), parameter :: par_a = 10.d0
    real(8), parameter :: par_b = 28.d0
    real(8), parameter :: par_c = 8.d0/3.d0
    real(8), parameter  :: dt=1.d-2
    real(8), parameter :: dt_Runge = 1.d-4
    real(8), parameter :: dx0 = 1.d0
    real(8), parameter :: dy0 = 4.d0
    real(8), parameter :: dz0 = 2.d0
    real(8), parameter :: x0 = 0.d0
    real(8), parameter :: y0 = 0.1d0
    real(8), parameter :: z0 = 0.d0
    integer, parameter :: lyapnov_step = 3000 !リヤプノフ指数計測回数
    integer, parameter :: tau =20
    integer, parameter :: rc_step = lyapnov_step * tau
    integer, parameter :: skip_step = 5000
    integer, parameter :: traning_step = 500
    integer, parameter :: sample_num   = 10
    integer, parameter :: tr_data_size = traning_step*sample_num
    integer, parameter :: epoch        = 20
    integer, parameter :: ly_epoch     = 1500
    integer, parameter :: god_step = 0
    real(8), parameter :: epsi0 =1.d-3
    
    integer, parameter :: ly_skip_step = 0
    integer, parameter :: NOW = 2
    integer, parameter :: BEFORE = 1
    integer, parameter :: in_node = 1
    integer, parameter :: out_node = 1
    real(8), parameter :: G0 = 1.d0
    real(8), parameter :: gusai0 = 0.d0
     real(8) :: PI=3.14159265358979
    real(8) w_out(rc_node,out_node)
    real(8) w_in (in_node,rc_node)
    real(8) W_rnn(rc_node,rc_node)

  contains
    subroutine march(r,dr,z,U_rc,S_rc,fstep,ftau,gusai,alpha,G)
        real(8) U_rc(:,:)
        real(8) S_rc(:,:)
        real(8) r(0:tau,1:rc_node)
        real(8) dr(0:tau,1:rc_node)
        real(8) z(0:tau,1:rc_node)
        real(8) u_tmp(1:in_node)
        real(8) r_tmp(1:rc_node)
        real(8) DF(rc_node, rc_node)
        real(8) wu(rc_node),wr(rc_node)
        real(8) wu2,wr2
        real(8) gusai,alpha,G
        integer i,j,k,f,fstep,ftau,step
!        write(*,*) dr_befor(1:par_node)
        wu=0.d0
        wr=0.d0
        do i=1,rc_node
        do j=1,in_node
            wu(i)=wu(i)+u_rc(fstep,j)*W_in(j,i)
        enddo
        enddo
        do i=1,rc_node
        do j=1,rc_node
            wr(i)=wr(i)+r(ftau-1,j)*W_rnn(j,i)
        enddo
        enddo
        do i=1,rc_node
            z(ftau,i) =G*( wr(i)+wu(i)+gusai )
            r(ftau,i)=tanh(z(ftau,i))
        enddo
        do i = 1,out_node
		    s_rc(ftau,i)  = 0.d0
		    do k = 1,rc_node
		        s_rc(ftau,i) = s_rc(ftau,i) + r(ftau,k) * w_out(k,i)
		    enddo
		enddo
        do j=1,rc_node
        do i=1,rc_node
            DF(j,i)=d_tanh(z(ftau-1,j) )*G*W_rnn(i,j)
        enddo
        enddo
        do i=1,rc_node
            dr(ftau,i) = 0.d0
            do k=1,rc_node
                dr(ftau,i) = dr(ftau,i) + dr(ftau-1,k)*DF(k,i)
            enddo
        enddo
    end subroutine march
    subroutine create_dataset(dataset,trmax)
        real(8) dataset(:,:)
        real(8) X(3,1:3)
        integer istep,trmax
        X(BEFORE,:)=1.d0
!        open(10,file='./data/output_Runge_Lorenz.dat')
        write(*,*) "=========================================="
        write(*,*) "START   CREATE TRANING SET"
        write(*,*) "    TOTAL STEP =",trmax
        write(*,*) "=========================================="
        do istep = 1, skip_step
            call Runge_Kutta_method(X(NOW,1:3),X(BEFORE,1),X(BEFORE,2),X(BEFORE,3),istep)
!            if(mod(istep,10) == 0) write(10,*) istep ,x(NOW,1),x(NOW,2),x(NOW,3)
!            write(10,*) istep ,x(NOW,1),x(NOW,2),x(NOW,3)
            x(BEFORE,1:3) = x(NOW,1:3)
        enddo
!        close(10)
        do istep=1,trmax
            call Runge_Kutta_method(X(NOW,1:3),x(BEFORE,1),x(BEFORE,2),x(BEFORE,3),istep)
            x(BEFORE,1:3) = X(NOW,1:3)
            dataset(istep,1:3)= X(NOW,1:3)
        enddo
        call standard(dataset,3,trmax)
    end subroutine create_dataset
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	subroutine standard(a,data_node,step)
		real(8) a(:,:)
		integer data_node,step
		real(8) a_mean(data_node),a_var(data_node)
		integer i,j
		do i = 1,data_node
			a_mean(i)=mean(a(1:step,i),step)
		enddo
		do i = 1,data_node
			a_var(i)=variance(a(1:step,i),a_mean(i),step)
		enddo
		do j = 1,data_node
			do i=1,step
				a(i,j)=(a(i,j)-a_mean(j))/a_var(j)
			enddo
		enddo
	end subroutine standard
	function mean(a,step) result(out)
		real(8), intent(in) :: a(:)
		real(8) :: out
		integer i,step
		out=0.d0
		do i=1,step
			out = out + a(i)
		enddo
		out= out/dble(step)
	end function mean
	function variance(a,a_mean,step) result(out)
		real(8), intent(in) :: a(:),a_mean
		real(8) :: out
		integer i,step
		out=0.d0
		do i=1,step
			out = out + (a(i)-a_mean)**2
		enddo
		out= (out/dble(step))**0.5
	end function variance
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!-----------------------------------------------------------------------------
!■ルンゲクッタ法での近似
!-----------------------------------------------------------------------------
	subroutine Runge_Kutta_method(r_now,x,y,z,step)
		real(8) x,y,z
		real(8) r_now(1,1:3)
		real(8) k1(1:3),k2(1:3),k3(1:3),k4(1:3)
		real(8) k(1:3)
		integer i,j,step
		
		
        do j=1, int(dt/dt_Runge)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		    k1(1)=f1(x,y,z)
		    k1(2)=f2(x,y,z)
		    k1(3)=f3(x,y,z)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		    k2(1)=f1(x+k1(1)*0.5d0 ,y+k1(2)*0.5d0 , z+k1(3)*0.5d0)
		    k2(2)=f2(x+k1(1)*0.5d0, y+k1(2)*0.5d0 , z+k1(3)*0.5d0)
		    k2(3)=f3(x+k1(1)*0.5d0, y+k1(2)*0.5d0 , z+k1(3)*0.5d0)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		    k3(1)=f1(x+k2(1)*0.5d0, y+k2(2)*0.5d0 , z+k2(3)*0.5d0)
		    k3(2)=f2(x+k2(1)*0.5d0, y+k2(2)*0.5d0 , z+k2(3)*0.5d0)
		    k3(3)=f3(x+k2(1)*0.5d0, y+k2(2)*0.5d0 , z+k2(3)*0.5d0)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		    k4(1)=f1(x+k3(1), y+k3(2) , z+k3(3))
		    k4(2)=f2(x+k3(1), y+k3(2) , z+k3(3))
		    k4(3)=f3(x+k3(1), y+k3(2) , z+k3(3))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		    do i=1,3
		    	k(i)=( k1(i) + 2.d0*k2(i) + 2.d0*k3(i) + k4(i) )/6.d0
		    enddo
		    x = x+k(1)
		    y = y+k(2)
		    z = z+k(3)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        enddo
		r_now(1,1) = x
		r_now(1,2) = y
		r_now(1,3) = z
!        open(20,file='output_001_Euler_x2y10z5.dat',position='append')
!        write(20,*) step ,r_now(1,1:3)
!        close(20)
    contains
        function f1(x,y,z)
	    	real(8) x,y,z
	    	real(8) f1
	    	f1=dt_Runge*(-par_a*(x-y))
	    end function f1
	    function f2(x,y,z)
	    	real(8) x,y,z
	    	real(8) f2
	    	f2=dt_Runge*((par_b-z)*x - y)
	    end function f2
	    function f3(x,y,z)
	    	real(8) x,y,z
	    	real(8) f3
	    	f3=dt_Runge*(x*y -par_c*z)
	    end function f3
	end subroutine Runge_Kutta_method
!======================================================
!======================================================

    subroutine Initialization_file
        open(10,file='./data/output_Runge_Lorenz.dat',status='replace')
        close(10)
    end subroutine Initialization_file

    subroutine Ly_Calculation(abs_dr,lyapnov,step)
        real(8),intent(inout) :: lyapnov
        real(8),intent(in   ) ::  abs_dr
        integer step

        lyapnov =(lyapnov*dble(step-1) + log(abs_dr) /tau)/dble(step)
100 format(i5,a,f15.10,f15.10)
    end subroutine Ly_Calculation
    function sigmoid(x)
        real(8) x, sigmoid   ! 順番は関係ありません
        sigmoid = 1.d0  / (1.d0 + exp(-x) )
    end function sigmoid
    function d_sigmoid(x)
        real(8) x, d_sigmoid   ! 順番は関係ありません
        d_sigmoid = sigmoid(x)*(1-sigmoid(x))
    end function d_sigmoid
    
    function d_tanh(x)
        real(8) x, d_tanh   ! 順番は関係ありません
        d_tanh = 4.d0 /(( exp(x)+exp(-x) )**2)
    end function d_tanh
    subroutine calu_absdr(abs_dr,dr,ftau)
        real(8) abs_dr
        real(8),intent(in):: dr(0:tau,1:rc_node)
        integer fi,ftau
        abs_dr=0.d0
        do fi=1,rc_node
!            write(*,*) fi,dr(ftau,fi)
            abs_dr = abs_dr+dr(ftau,fi)**2
        enddo
        abs_dr=abs_dr**0.5d0
    end subroutine calu_absdr
    subroutine calu_ERR(S_rc_out,S_rc_data,ERR)
        real(8) S_rc_data(1:tau,1:out_node)
        real(8) S_rc_out(1:tau,1:out_node)
        real(8) ERR
        integer fi,fj,fk
        ERR=0.d0
        do fi=1,tau
            do fj=1,out_node
                ERR=ERR+(S_rc_data(fi,fj)-S_rc_out(fi,fj))**2
            enddo
        enddo
        ERR=ERR/dble(tau)
    end subroutine calu_ERR
    function rand_normal(mu,sigma)
    	real(8) mu,sigma
    	real(8) rand_normal
    	real(8) z,p1,p2
    	call random_number(p1)
    	call random_number(p2)
    	!write(*,*) p1,p2
        z=sqrt( -2.0*log(p1) ) * sin( 2.0*PI*p2 );
        rand_normal= mu + sigma*z
    end function rand_normal
    
end module lyapnov

program main
use lyapnov
  implicit none
  include 'mpif.h'
  integer myrank
  integer procs !並列数
  integer ierr  !エラーコード
  
    real(8) U_tr(tr_data_size,in_node) !今は一次元、列サイズはトレーニング時間
    real(8) S_tr(tr_data_size,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8) U_rc(rc_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8) S_rc(rc_step,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8) S_rc_out(tau,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8) dr0(rc_node), lyap,abs_dr,abs_dr2
    real(8) r_before(1:rc_node)
    real(8) DF(rc_node, rc_node)
    real(8) dataset(1:rc_step+tr_data_size+god_step+100,par_node)
    real(8) r(0:tau,1:rc_node)
    real(8) dr(0:tau,1:rc_node)
    real(8) z(0:tau,1:rc_node)
    real(8) G_tmp,av_degree,p,ERR,ERR_ave,iepsi,err_tr,lyap_ini
    integer now_step,epsi_adjust,itraning_step,isample_num
    integer istep,itau,iG,iepoch,j,k,i,n,ilyap_err
    character(6) :: cist,cist2
    call random_seed
!    open(50,file="./data_out/lyap_err_1000.dat",status='replace')
!    close(50)
    
    do iepoch=1,1500
        write(cist,'(i4.4)') iepoch
        open(50,file='./data_renban/lyap-err_epoch.'//cist,status='replace')
        close(50)
    enddo
        
    !    open(41,file="./data_out/lyapnov_end.dat",status='replace')
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !■create_dataset
    !-------------------------------------------------
    call create_dataset(dataset,rc_step+tr_data_size+god_step+100)
    U_tr(1:tr_data_size,1 )=dataset(1:tr_data_size,1)
    s_tr(1:tr_data_size,1)=dataset(1+god_step:tr_data_size+god_step,2 )
    U_rc(1:rc_step,1 )=dataset(1+tr_data_size:tr_data_size+rc_step,1)
    S_rc(1:rc_step,1)=dataset(1+tr_data_size+god_step:tr_data_size+rc_step+god_step,2)
    !    open(10,file='./data/output_Runge_Lorenz.csv')
    !    do i=1,12000
    !        write(10,*) dataset(i,1),",",dataset(i,2),",",dataset(i,3)
    !    enddo
    !    close(10)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!K
    call MPI_Init(ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, procs, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
    
    do ilyap_err =1,5000
    if(mod(ilyap_err,procs)==myrank) then
    !	do isample_num=100,500,100
        isample_num=sample_num
        call random_seed
        !write(cist2,'(i5.5)') ilyap_err
        !open(41,file='./data_renban/lyapnov_end.'//cist2,status='replace')
        itraning_step=int(dble(tr_data_size)/dble(isample_num) )
        
        do i=1,rc_node
        do k=1,rc_node
            w_rnn(i,k)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
        enddo
        enddo
        do i=1,in_node
        do k=1,rc_node
            w_in(i,k)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
        enddo
        enddo
        do i=1,rc_node
        do k=1,out_node
            w_out(i,k)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
        enddo
        enddo
    	iepsi=epsi0
    	
    
        do iepoch=1,ly_epoch
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !■初期化
    !-------------------------------------------------
        lyap=0.d0
        r=1.d0
        abs_dr=1.d0
        ERR_ave=0.d0
        write(cist,'(i4.4)') iepoch
        call Initialization_file
        call random_number(dr(0,:))
        call calu_absdr(abs_dr,dr,0)
        dr(0,:) = dr(0,:)/ abs_dr
        abs_dr=1.d0
        
        write(cist,'(i4.4)') iepoch
!        open(43,file='./data_renban/lyapnov.'//cist,status='replace')
!        open(51,file='./data_renban2/rc_out.'//cist,status='replace')
    !■Wout calu
    !-------------------------------------------------
    !    call calu_wout(U_tr,S_tr)
    !    call rc_traning_own_fortran(in_node,out_node,rc_node,traning_step,tau,gusai0,1.d0,G0,&
    !                    u_tr,s_tr,U_rc,w_in,w_rnn,w_out)
                        
        call rnn_traning_own_fortran(in_node,out_node,rc_node,traning_step,rc_step,&
                        isample_num,epoch,iepsi,g0,gusai0,&
                        u_tr,s_tr,u_rc,w_out,w_rnn,w_in,err_tr)
    
    !    write(*,*) "=========================================="
    !    write(*,*) "START   CALUCULATE LY"
    !    write(*,*) "=========================================="
    !    dr(0,:) =1.d0
    !    write(*,*) dr(0,:)
    !    call calu_absdr(abs_dr,dr,0)
    
        do istep = 1, lyapnov_step
            dr(0,:) = dr(0,:)/ abs_dr
            
            do itau = 1,tau
                now_step = (istep-1)*tau +itau
                call march(r,dr,z,U_rc,S_rc_out,now_step,itau ,gusai0,1.d0,G0)
                call calu_absdr(abs_dr,dr,itau)
            enddo
            call calu_absdr(abs_dr,dr,tau)
            call Ly_Calculation(abs_dr,lyap,istep)
            
            call calu_ERR(S_rc_out(1:tau,1:out_node),S_rc((istep-1)*tau+1 : istep*tau , 1:out_node),ERR)
            
            ERR_ave=(ERR_ave*dble(istep-1)+ERR)/dble(istep)
    
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*) "---------------------------------------------------------"
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  iepoch
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  ilyap_err
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  istep ,int(istep*100/lyapnov_step),'%'
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  "abs_dr           ====", abs_dr
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  "lyapnov          ====", lyap
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  "ERR              ====", ERR_ave**0.5d0
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  "err_tr           ====", err_tr
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  "lyapnov_epoch    ====", iepoch
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  "epsi             ====", iepsi
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  "isample          ====", isample_num
!            if(mod(istep,nint(lyapnov_step*1.d0))==0) write(*,*)  "itraning_step    ====", itraning_step
    
            r(0,1:rc_node)  = r(tau,1:rc_node)
            dr(0,1:rc_node) = dr(tau,1:rc_node)
 !           write(43,*) istep,lyap,abs_dr
 !           do i=1,tau
 !               if(istep>lyapnov_step-15) write(51,*) S_rc_out(i,1), S_rc((istep-1)*tau+i,1),U_rc((istep-1)*tau+i,1)
 !           enddo
        enddo
        if(iepoch==1) lyap_ini=lyap
        open(50,file='./data_renban/lyap-err_epoch.'//cist,position='append')
        write(50,"(4e14.6)") lyap ,err_tr,ERR_ave**0.5d0,lyap_ini
        close(50)
        enddo
    endif
    enddo
    call MPI_Finalize(ierr)
200 format(a,f15.10)
end program