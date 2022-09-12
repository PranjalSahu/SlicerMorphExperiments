from ast import Raise
import os
import glob

data_path = '/data/Apedata/CorrectData/data/'
animal_type = 'Mus'
write_path = "/data/Apedata/Slicer-cli-outputs/"
script_path = "/data/SlicerMorph/slicermorphextension/ITKALPACA/ITKALPACA/ITKALPACA.py"

# Gorilla cases
allnames_gorilla = [
    "USNM582726",
    "USNM590951",
    "USNM599165",
    "USNM590953",
    "USNM252577",
    "USNM590947",
    "USNM599167",
    "USNM176217",
    "USNM297857",
    "USNM252575",
    "USNM174715",
    "USNM220324",
    "USNM176216",
    "USNM590954",
    "USNM252580",
    "USNM220060",
    "USNM252578",
    "USNM176209",
    "USNM599166",
    "USNM590942",
    "USNM174722",
]

# Pan Cases
allnames_pan = [
    "USNM176228",
    "USNM174701",
    "USNM084655",
    "USNM174704",
    "USNM174707",
    "USNM174703",
    "USNM220062",
    "USNM174710",
    "USNM220065",
    "USNM176236",
]


# Pongo cases
allnames_pongo = [
    "USNM153830",
    "USNM145302",
    "USNM142189",
    "USNM142188",
    "USNM145308",
    "USNM153805",
    "USNM142194",
    "USNM197664",
    "USNM145303",
    "USNM399047",
    "USNM153822",
    "USNM142185",
    "USNM145309",
    "USNM153824",
    "USNM145307",
    "USNM145300",
    "USNM153806",
]

# Mus cases
allnames_mus = [
    "C57BL6_J_",
    "C57BL_10J_",
    "C57BLKS.J_",
    "C57BL_6NJ_",
    "AKR_J_",
    "DBA_1J_",
    "CAST_EIJ_",
    "BALB_CJ_",
    "NOR.LtJ_",
    "A_J_",
    "NZB.BINJ_",
    "B6D2F1_J_",
    "PL.J_",
    "DBA_2J_",
    "B6CBAF1.J_",
    "SF.CamEiJ_",
    "MOLG.DnJ_",
    "LP.J_",
    "129S1_SVIMJ_",
    "PWK.PhJ_",
    "KK.HlJ_",
    "SWR.J_",
    "B6129PF1.J_",
    "SPRET.EiJ_",
    "C3H_HEJ_",
    "C57L_J_",
    "B6C3F1.J_",
    "129X1_SVJ_",
    "SJL_J_",
    "B6AF1_J_",
    "B6129SF1.J_",
    "NOD_SHILTJ_",
    "B6SJLF1.J_",
    "C3H_HEOUJ_",
    "NZW_LACJ_",
    "NZBWF1_J_",
    "X129P3.J_",
    "CBA_CAJ_",
    "TALLYHO_JNGJ_",
    "BTBR_T_Itpr3tf_j_",
    "CBA_J_",
    "CB6F1_J_",
    "LG.J_",
    "BALB_CBYJ_",
    "C3HeB.FeJ_",
    "PERC.EiJ_",
    "FVB_NJ_",
    "MRL_MPJ_",
    "CAF1_J_",
    "NU_J_",
    "NZO.HlLtJ_"
]
# MOVING FILE i.e. TEMPLATE FILE IS CONSTANT


if animal_type == 'Gorilla':
    template_mesh = data_path + "Gorilla/meshes/USNM176211-Cranium.ply"
    template_landmark = data_path + "Gorilla/landmarks/USNM176211_LM1.fcsv"
    allnames = allnames_gorilla
elif animal_type == "Pan":
    template_mesh = data_path + "Pan/meshes/USNM220063-Cranium_merged_1.ply"
    template_landmark = data_path + "Pan/landmarks/USNM220063_LM1.fcsv"
    allnames = allnames_pan
elif animal_type == "Pongo":
    template_mesh = data_path + "Pongo/meshes/USNM588109-Cranium.ply"
    template_landmark = data_path + "Pongo/landmarks/USNM588109_LM1.fcsv"
    allnames = allnames_pongo
elif animal_type == 'Mus':
    template_mesh = data_path + "Mus/template/template.ply"
    template_landmark = data_path + "Mus/template/template.fcsv"
    allnames = allnames_mus
else:
    Raise("Animal Type not present")

for name in allnames:
    print("====================================================")
    print("Registering ", name)
    meshfile = glob.glob(
        data_path + animal_type + "/meshes/" + name + "*.ply"
    )[0]

    landmarkfile = glob.glob(
        data_path
        + animal_type
        + "/landmarks/"
        + name
        + "*.fcsv"
    )[0]

    os.system(
        "python " + script_path + " "
        + meshfile
        + " "
        + template_mesh
        + " 0.4 100 0.3 2 100 2 10 5 100 2000 2 2 "
        + template_landmark
        + " 1"
    )
