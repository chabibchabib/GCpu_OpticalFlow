#!/usr/bin/env python3
import cv2
import sys

def parameters_func(tab,parameters):
    '''
    parameters_func will associate the parameters and values given in tab with their 
    their correspondant fields in the dico of parameters 
    Parameters: 
        tab: a table of strings containing the keys and their values (Example: lmbda=0.05)
        parameters: the dictionary of parameters 
    returns: 
         will associate the parameters and values given in tab with their 
        their correspondant fields in parameters
    '''
    #The parameters of type int
    tabint=["pyram_levels","ordre_inter","size_median_filter","max_linear_iter","max_iter"]
    #The parameters of type float
    tabfloat=["factor","lmbda","lambda2","lambda3"]
    if len(tab)>4:
        for i in range(4,len(tab)):

            #Index of '='
            idx= tab[i].find("=")
            #The name of the parameter (key)
            key=tab[i][:idx]
            #The value of the parameter
            value=tab[i][idx+1:len(tab[i])]
            # Modify the value of the parameter 

            if key in tabint:
                parameters[key]=int(value)
            elif key in tabfloat:
                parameters[key]=float(value)
            elif(key=="Mask"):
                existP=0
                #Find the Path_Mask 
                for j in range(4,len(tab)):
                    idxP= tab[j].find("=")
                    keyP=tab[j][:idxP]
                    if keyP=="Path_Mask":
                        valueP=tab[j][idxP+1:len(tab[j])]
                        Mask=valueP+value
                        existP=1
                if(existP==0):
                    raise ValueError("No founded path for Mask")
                #Read the image Mask if exist 
                parameters[key]=cv2.imread(Mask,0)


                    
            elif((key not in tabint) and(key not in tabfloat) and key !="Mask" and key!="Path_Mask" ):
                # No founded Key 
                raise ValueError("Problem with the parameter",key)
    if len(tab)<4:
        # At least we must have 4 parameters: The name of the main file, 2 Images and Their Path 
        raise ValueError("Not enough parameters")

'''
# Test the function 
parameters = {"pyram_levels": 3, "factor": 1/0.5, "ordre_inter": 3, "size_median_filter": 5, "max_linear_iter": 1, "max_iter": 10,
              "lmbda": 3.*10**4, "lambda2": 0.001, "lambda3": 1., "Mask": None}

parameters_func(sys.argv,parameters)
print(parameters["Mask"])  '''       
def find(string,parameters,Mask=''):
    ''' find function will help the user to modify the value of a specified parameter 
    Parameters:
        -string: a string contains the name and the value the parameter to modify 
        (example:lmbda=2e+02)
        -parameters: the dictionary of parameters
    Returns:
        No return but it will modify the desired parameter      
    '''

    #The parameters of type int
    tabint=["pyram_levels","ordre_inter","size_median_filter","max_linear_iter","max_iter"]
    #The parameters of type float
    tabfloat=["factor","lmbda","lambda2","lambda3"]
    #Tab of Mask
    tabmask=["Mask","Mask_Path"]
    #Index of '='
    idx= string.find("=")
    #The name of the parameter (key)
    key=string[:idx]
    #The value of the parameter
    value=string[idx+1:len(string)]
    # Modify the value of the parameter 

    if key in tabint:
        parameters[key]=int(value)
    elif key in tabfloat:
        parameters[key]=float(value)
   
    elif((key not in tabint) and(key not in tabfloat) and (key not in tabmask) ):

        raise ValueError("Problem with the parameter",key)

def replace_main(tab,parameters):
    '''Replace the dictionary of parameters fields with the srings of tab array
        - Parameters:
            -tab: Table of strings 
            -parameters: the dictionary of parameters
         '''
    if len(tab)>4:
        for i in range(4,len(tab)):
            find(tab[i],parameters)

    if len(tab)<4:
            raise ValueError("Not enough parameters")
