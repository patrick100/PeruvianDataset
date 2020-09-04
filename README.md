# PeruvianDataset
Pre-processing scripts of Peruvian Dataset

# Clasificacion

## 1. poner los objs en la carpeta OBJs y crear la carpeta destino Peruvian-DB
## 2. ejecutar classify.py para ordenar la db en categorias divididas en train y test.


## Normalizacion

Para normalizar es necesario que este clasificada la db y seguir el siguiente orden convertir los objs a off para despues convertirlos a ply y despues a objs 
 ejecutar OFFtoH5.py antes de ejecutar dar permisos a meshconv (es necesario editar el codigo y crear la carpeta destino previamente)
 
### sudo chmod u+x meshconv
