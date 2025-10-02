add_library(fabm_models_bsh OBJECT
            bsh_model_library.F90
	    ergom_neccton.F90
           )

add_dependencies(fabm_models_bsh fabm_base)
