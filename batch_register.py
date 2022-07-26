import os
import glob

# Gorilla cases
allnames = [
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

#allnames = ["USNM590954"]
# # bad result cases
# badcases = ['USNM582726']
# # failed cases
# allnames = ["USNM176209", "USNM220324", "USNM582726"]

# Pan Cases
# allnames = [
#     "USNM176228",
#     "USNM174701",
#     "USNM084655",
#     "USNM174704",
#     "USNM174707",
#     "USNM174703",
#     "USNM220062",
#     "USNM174710",
#     "USNM220065",
#     "USNM176236",
# ]


# Pongo cases
# allnames = [
#     "USNM153830",
#     "USNM145302",
#     "USNM142189",
#     "USNM142188",
#     "USNM145308",
#     "USNM153805",
#     "USNM142194",
#     "USNM197664",
#     "USNM145303",
#     "USNM399047",
#     "USNM153822",
#     "USNM142185",
#     "USNM145309",
#     "USNM153824",
#     "USNM145307",
#     "USNM145300",
#     "USNM153806",
# ]
# MOVING FILE i.e. TEMPLATE FILE IS CONSTANT

template_mesh = "/data/Apedata/CorrectData/data/Gorilla/meshes/USNM176211-Cranium.ply"
template_landmark = "/data/Apedata/CorrectData/data/Gorilla/landmarks/USNM176211_LM1.fcsv"
animal_type = 'Gorilla'

# template_mesh = "/data/Apedata/CorrectData/data/Pan/meshes/USNM220063-Cranium_merged_1.ply"
# template_landmark = "/data/Apedata/CorrectData/data/Pan/landmarks/USNM220063_LM1.fcsv"
# animal_type = "Pan"

#template_mesh = "/data/Apedata/CorrectData/data/Pongo/meshes/USNM588109-Cranium.ply"
#template_landmark = "/data/Apedata/CorrectData/data/Pongo/landmarks/USNM588109_LM1.fcsv"
#animal_type = "Pongo"

write_path = "/data/Apedata/Outputs_RANSAC/"

for name in allnames:
    print("====================================================")
    print("Registering ", name)
    meshfile = glob.glob(
        "/data/Apedata/CorrectData/data/" + animal_type + "/meshes/" + name + "*.ply"
    )[0]

    landmarkfile = glob.glob(
        "/data/Apedata/CorrectData/data/"
        + animal_type
        + "/landmarks/"
        + name
        + "*.fcsv"
    )[0]

    os.system(
        "time python script_FPFH_RANSAC_Deform.py "
        + meshfile
        + " "
        + landmarkfile
        + " "
        + template_mesh
        + " "
        + template_landmark
        + " "
        + write_path
    )
