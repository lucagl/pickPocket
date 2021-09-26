# pickPocket
An ad-hoc hierarchical clustering algorithm able to extract and rank pockets. Extraction is based on geometrical primitives  generated by the NanoShaper molecular surface software. The ranking is based on Isolation Forest anomaly detector.
## Details:
### Clustering:
This script performs a hierarchical merging of "(regular) spherical probes" extracted from several calls to the NanoShaper software. NanoShaper is called externally by the script. The clustering process resemble to a single-linkage clustering in the sense that only distances between single probes matter rather than global aggregate information such as the center of mass. However, special rules are present to perform such aggregation.
### Ranking:
Ranking is based on Isolation Forest (IF) anomaly detector. IF is provided as a scikit-learn object previously trained and loaded from a  provided binary file (within)

## Requirements:
 - The NanoShaper executable is provided and must be installed has descibed in "NS_installer"
 - (Reccomended) Recompile locally the shared library. Instructions and source code in "C_tools"

## Instructions:

### Simple (use directly an executable)
python3 -m pickPocket <file.pqr>

TODO: more like output produced etc..

### Advanced
Extra set up files: config and input files.. Explained below

Examples of advanced use are provided into the scripts folder with related advanced input file

**The input file**

**The config file**
