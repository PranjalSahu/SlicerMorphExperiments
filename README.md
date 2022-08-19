## What is this project ?

This project is **NOT** the official Slicer repository. It is a fork of the SlicerMorph repository hosted at https://github.com/SlicerMorph/SlicerMorph.

It is used to organize experiments, coordinate work done by the [KitwareMedical][KitwareMedical] team and develop modules that will eventually be contributed back to the official repository.

The overall goal is to ultimately help with:

> The integration of maintainable and effective methods for point cloud alignment, i.e. registration, of large-scale SlicerMorph datasets.
>
> This includes the integration and extension of Insight Toolkit (ITK) registration methods, and Visualization Toolkit (VTK) point cloud filtering and transformation methods, as well as the testing & evaluation of the effectiveness of these methods on sample SlicerMorph datasets.

[KitwareMedical]: https://www.kitware.com/teams/medical-computing

```
# Install Dependencies using
# /home/pranjal.sahu/Downloads/Slicer-5.0.3-linux-amd64/bin/PythonSlicer -m pip install --prefix=/data/SlicerMorph/ITKALPACA-python-dependencies itk==5.3rc4
# python -m pip install -U --no-deps --prefix=/data/SlicerMorph/ITKALPACA-python-dependencies /data/SlicerMorph/LinuxWheel39_fpfh_5.3rc4_again/itk_fpfh-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-cache-dir

```
Command to run the extension with Slicer
```
./Slicer --additional-module-path /data/SlicerMorph/slicermorphextension/ITKALPACA/ITKALPACA/
```

All the result outputs are written at this path:
```
WRITE_PATH = "/data/Apedata/Slicer-cli-outputs/"
```

## Frequently asked questions

### What define an experiment and where are they organized ?

Experiments may be contributed as scripts or Jupyter notebooks.

They are organized in folders associated with the orphan branch named [kitware-experiments][].

[kitware-experiments]: https://github.com/KitwareMedical/SlicerMorph/tree/kitware-experiments

### How to contribute improved SlicerMorph modules ?

Topics should be based of the [master][] branch and may be contributed as pull requests created against the upstream SlicerMorph repository.

[master]: https://github.com/KitwareMedical/SlicerMorph/tree/master

### Are the sample datasets publicly available ?

See https://github.com/SlicerMorph/Mouse_Models


### How to ask questions specific to this effort ?

Visit [Slicer forum](https://discourse.slicer.org) and send a direct message to the following recipients:

| FirstName LastName | Discourse Handle |
|--|--|
| Murat Maga | [muratmaga][] |
| Sara Rolfe | [smrolfe][] |
| Arthur Porto | [agporto][] |
| Matt McCormick | [thewtex][] |
| Pranjal Sahu | [pranjal.sahu][] |
| Jean-Christophe Fillion-Robin | [jcfr][] |

_For easy copy-paste: `muratmaga`, `smrolfe`, `agporto`, `thewtex`, `pranjal.sahu`, `jcfr`_


[muratmaga]: https://discourse.slicer.org/u/muratmaga
[smrolfe]: https://discourse.slicer.org/u/smrolfe
[agporto]: https://discourse.slicer.org/u/agporto
[thewtex]: https://discourse.slicer.org/u/thewtex
[pranjal.sahu]: https://discourse.slicer.org/u/pranjal.sahu
[jcfr]: https://discourse.slicer.org/u/jcfr

## License

This software is licensed under the terms of the [BSD 2-Clause License](LICENSE.md).

