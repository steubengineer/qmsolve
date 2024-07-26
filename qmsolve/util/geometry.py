# used for loading and manipulating geometry from .stl mesh flies. Requires pymeshlab

import pymeshlab as ml
import numpy as np
from qmsolve import Å

def load_geometry( filename ):
    '''
    Loads the specified file into a mesh
    :param meshfile: filename of the input mesh stl.

    :Note: stl files have no units.  We will ASSUME that the mesh is saved in units of Å.
    This means you can make a model in whatever scale you like. If you make a feature 100mm long,
    save the stl, and load it here, it will be 100Å long.
    '''

    #slightly idiomatic - load the mesh into a mesh set, only to pull it back out later
    meshes = ml.MeshSet()
    meshes.load_new_mesh( filename )
    meshes.set_current_mesh( 0 )

    #proably ought to do some error handling for a bad file...

    # check that the mesh is manifold
    topos = meshes.get_topological_measures()
    manifoldq = topos['is_mesh_two_manifold']
    if manifoldq:
        print( 'loaded 2-manifold mesh of', topos['faces_number'],'faces and', topos['vertices_number'], 'vertices')
    else:
        print( 'WARNING: loaded mesh is not 2-manifold!' )

    #do some very basic mesh cleanup
    #meshes.meshing_merge_close_vertices( threshold=ml.PercentageValue(0.1) ) #merge supernearby vertices
    #meshes.meshing_remove_unreferenced_vertices()

    #scale to Å
    inputmesh = meshes.current_mesh()
    inputverts = inputmesh.vertex_matrix()
    inputfaces = inputmesh.face_matrix()

    scaledverts = inputverts*Å
    scaledmesh = ml.Mesh( vertex_matrix = scaledverts, face_matrix = inputfaces )

    return scaledmesh

def pointmesh( points ):
    pointlist = np.transpose((points.x.flatten(), points.y.flatten(), points.z.flatten()))
    pointmesh = ml.Mesh(pointlist)
    print('built pointmesh of',pointlist.shape[0],'points')

    return pointmesh

def compute_sdf( meshfile, points ):
    '''
    Computes the Signed Distance Function
    :param meshfile: filename of the input mesh stl. The distance relative to this mesh is computed
    :param points: points used by Hamiltonian at which the SDF is to be calculated
    :return: the distance of every hamiltonian point to the input meshfile's STL mesh
    '''

    #TODO: the PercentageValue in compute_scalar_by_dist... needs to be user settable

    meshes = ml.MeshSet()
    meshes.add_mesh( load_geometry(meshfile) )
    meshes.set_current_mesh( 0 )

    meshes.add_mesh( pointmesh(points) )
    meshes.set_current_mesh( 1 )

    meshes.compute_scalar_by_distance_from_another_mesh_per_vertex(measuremesh=1, refmesh=0,
                                                                   signeddist=True,
                                                                   maxdist=ml.PercentageValue(10.0))

    distances = meshes.current_mesh().vertex_scalar_array()
    distmat = np.reshape( distances, points.x.shape)
    return distmat


