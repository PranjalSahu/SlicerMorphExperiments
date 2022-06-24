import os
import glob

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

# bad result cases
badcases = ['USNM582726']

# failed cases
allnames = ["USNM176209", "USNM220324", "USNM582726"]

# Moving file i.e. template file is constant
for name in allnames:
    print("====================================================")
    print("Registering ", name)
    meshfile = glob.glob("/data/Apedata/CorrectData/data/Gorilla/meshes/" + name + "*.ply")[0]
    landmarkfile = glob.glob(
        "/data/Apedata/CorrectData/data/Gorilla/landmarks/" + name + "*.fcsv")[0]

    os.system(
        "time python script_RegisterEndToEnd.py "
        + meshfile
        + " "
        + landmarkfile
        + " /data/Apedata/CorrectData/data/Gorilla/meshes/USNM176211-Cranium.ply /data/Apedata/CorrectData/data/Gorilla/landmarks/USNM176211_LM1.fcsv"
    )
