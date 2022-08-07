#!/usr/bin/env python-real

import sys
import script_FPFH_RANSAC_Deform as alpaca


def main(
    target,
    source,
    subsample_radius,
    number_of_ransac_points,
    inlier_value,
    fpfh_radius,
    fpfh_neighbors,
    deform_sigma,
    deform_neighbourhood,
    bspline_grid,
    deformable_iterations
):
    alpaca.process(
        target,
        source,
        subsample_radius,
        number_of_ransac_points,
        inlier_value,
        fpfh_radius,
        fpfh_neighbors,
        deform_sigma,
        deform_neighbourhood,
        bspline_grid,
        deformable_iterations
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: ITKALPACA <target> <source> <subsample_radius> <number_of_ransac_points> <inlier_value> <fpfh_radius> <fpfh_neighbors> <deform_sigma> <deform_neighbourhood> <bspline_grid> <deformable_iterations>"
        )
        sys.exit(1)
    print("Parameters are ")
    print("target : ", sys.argv[1])
    print("source : ", sys.argv[2])
    print("subsample_radius : ", sys.argv[3])
    print("number_of_ransac_points : ", int(sys.argv[4]))
    print("inlier_value : ", float(sys.argv[5]))
    print("fpfh_radius : ", sys.argv[6])
    print("fpfh_neighbors : ", sys.argv[7])
    print("deform_sigma : ", sys.argv[8])
    print("deform_neighbourhood : ", sys.argv[9])
    print("bspline_grid : ", int(sys.argv[10]))
    print("deformable_iterations : ", int(sys.argv[11]))

    main(
        target=sys.argv[1],
        source=sys.argv[2],
        subsample_radius=int(sys.argv[3]),
        number_of_ransac_points=int(sys.argv[4]),
        inlier_value=float(sys.argv[5]),
        fpfh_radius=int(sys.argv[6]),
        fpfh_neighbors=int(sys.argv[7]),
        deform_sigma=float(sys.argv[8]),
        deform_neighbourhood=int(sys.argv[9]),
        bspline_grid=int(sys.argv[10]),
        deformable_iterations=int(sys.argv[11])
    )