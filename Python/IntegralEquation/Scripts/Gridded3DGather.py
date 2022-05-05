"""
Extracts and grids a 3D receiver gather from the
Cardamom data

@author: Joseph Jennings
@version: 2020.05.19
"""
import sys, os, argparse, configparser
import seppy
# from gridhelper import plot_sxrx_geometry
import numpy as np
import subprocess

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("-nodeid",help="Node ID number",type=int,required=True)
ioArgs.add_argument("-filenum",help="File number",type=int,required=True)
# Gridding arguments
grdArgs = parser.add_argument_group('Gridding parameters')
grdArgs.add_argument("-bg1",help="Beginning of grid along first axis",type=float,required=True)
grdArgs.add_argument("-eg1",help="Ending of grid along first axis",type=float,required=True)
grdArgs.add_argument("-dg1",help="Sampling of grid along first axis [0.05]",type=float)
grdArgs.add_argument("-bg2",help="Beginning of grid along second axis",type=float,required=True)
grdArgs.add_argument("-eg2",help="Ending of grid along second axis",type=float,required=True)
grdArgs.add_argument("-dg2",help="Sampling of grid along second axis [0.05]",type=float)
# Optional arguments
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
parser.add_argument("-gridqc",help="Plot the shots on a grid",type=str)
parser.add_argument("-min1",help="Min1 for plotting [0.0]",type=float)
parser.add_argument("-max1",help="Max1 for plotting [30.1]",type=float)
parser.add_argument("-min2",help="Min2 for plotting [0.0]",type=float)
parser.add_argument("-max2",help="Max2 for plotting [30.1]",type=float)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
gridqc = sep.yn2zoo(args.gridqc)

# First get the file
path = '/data2/Cardamom/CARD_HYDRO/910.00-hydrophone_sgy/'
filein = path + '_910.00-hydrophone_sgy-%d.H'%(args.filenum)

# Window the file based on the node ID
ndid = args.nodeid
win1 = "Window_key synch=1 key1='nodeid' mink1=%d maxk1=%d verb=1 < %s \
    > node%d.H hff=node%d.H@@"%(ndid,ndid,filein,ndid,ndid)
if(verb): print(win1)
sp = subprocess.check_call(win1,shell=True)

# Perform Headermath
hmath = "Headermath key1='sx' eqn1='sx/1000.0-199.606' key2='sy' \
    eqn2='sy/1000.0-34.470' key3='rx' eqn3='rx/1000.0-199.606' \
    key4='ry' eqn4='ry/1000.0-34.470' < node%d.H > node%dhmath.H"%(ndid,ndid)
if(verb): print(hmath)
sp = subprocess.check_call(hmath,shell=True)

# Get only one side of the gun array (I think port?)
wingun = "Window_key synch=1 nkeys=1 key1='gunarray_p_s' \
    mink=1 maxk=1 verb=1 < node%dhmath.H > node%dhmathgun1.H"%(ndid,ndid)
if(verb): print(wingun)
sp = subprocess.check_call(wingun,shell=True)

# Get grid boundaries
bg1 = args.bg1; eg1 = args.eg1; dg1 = args.dg1
bg2 = args.bg2; eg2 = args.eg2; dg2 = args.dg2

# Window shots based on the grid
winsxsy =  "Window_key synch=1 nkeys=1 key1='sx' mink1=%f maxk1=%f \
    key2='sy' mink2=%f maxk2=%f verb=1 < node%dhmathgun1.H > node%dpatch3d.H"%(bg1,eg1,bg2,eg2,ndid,ndid)
if(verb): print(winsxsy)
sp = subprocess.check_call(winsxsy,shell=True)

# # QC the grid
# if(gridqc):
#   min1 = args.min1; max1 = args.max1
#   min2 = args.min2; max2 = args.max2
#   plot_sxrx_geometry("node%dhmathgun1.H"%ndid,"dslcconvpad.H",bg1,eg1,dg1,bg2,eg2,dg2,min1,max1,min2,max2,show=False)
#   plot_sxrx_geometry("node%dpatch3d.H"%ndid,"dslcconvpad.H",bg1,eg1,dg1,bg2,eg2,dg2,min1,max1,min2,max2,show=True)

# Compute the grid parameters
ng1 = int((eg1-bg1)/dg1+1)
ng2 = int((eg2-bg2)/dg2+1)

# Grid the sorted data
sort = "Sort3d copy_data=1 nkeys=2 key1=sx key2=sy verb=1 \
    ng1=%d og1=%f dg1=%f ng2=%d og2=%f dg2=%f \
    < node%dpatch3d.H > node%dgrid3d.H "%(ng1,bg1,dg1,ng2,bg2,dg2,ndid,ndid)
if(verb): print(sort)
sp = subprocess.check_call(sort,shell=True)

# Stack over the traces per bin
stack = "Stack axis2=2 < node%dgrid3d.H > node%dgrid3dstk.H"%(ndid,ndid)
if(verb): print(stack)
sp = subprocess.check_call(stack,shell=True)
