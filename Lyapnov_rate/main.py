import pandas as pd


#dataname_in = "lyap_initial01.csv"
dataname_in = "./data_renban/lyap-err_epoch."
#dataname_out = "./data_renban3/lyap_inital_from_python.dat"
dataname_out = "./data_renban3/lyap_epoch_from_python."
Xmin = -1.0
Xmax = 0.0
Ymin = -1.0
Ymax = 0.0
dx = 1.0
dy = 1.0


def get_data(istep):
    filename = dataname_in + str(istep).zfill(4)
    Dataf = pd.read_csv(filename ,
                        names=[1,2,3,4],
                        header =None,
                        sep='\s+'
                        #usecolumn
                        )
#    Dataf=Dataf.dropna(how='any')
    Dataf= Dataf.fillna(1000)
    #Dataf = Dataf.iloc[0:4]
    return Dataf
    
#def split_data(Dataf):
    

def output_data(Dataf,istep):
    filename=dataname_out + str(istep).zfill(4)
    print(filename)
    Dataf.to_csv(filename,
                    header=False,
                    index=False,
                    sep='\t',
#                    float_format='%6'
                    )



if __name__ == '__main__':
#    while(i<25):
#        data[i] = pd.read_csv('1 o_%03d.csv'%i)
#        i+=1
    for istep in range(0,1501):
        print(istep)
        Dataframe=get_data(istep)
#        print(Dataframe)
        output_data(Dataframe,istep)
        data_len0=len(Dataframe)
#        print(data_len0)
#    Split_Data(Dataframe)
