#%%
#
# Solution mrotat
#
from __future__ import print_function
#
# SpiceyPy package:
#
import spiceypy
import os

def mrotat():
    #
    # Local parameters
    #
    METAKR =os.path.join(os.path.dirname(__file__), 'mrotate.tm')

    #
    # Load the kernels that this program requires.
    #
    spiceypy.furnsh( METAKR )

    #
    # Convert our UTC string to seconds past J2000 TDB.
    #
    timstr = '2007 JAN 1 00:00:00'
    et     = spiceypy.str2et( timstr )

    rotmat = spiceypy.pxform("J2000", "IAU_EARTH", et)
    rotated = 

    #
    # Look up the apparent position of the Earth relative
    # to the Moon's center in the IAU_MOON frame at ET.
    #
    [imoonv, ltime] = spiceypy.spkpos(
        'earth', et, 'iau_moon', 'lt+s', 'moon' )

    #
    #Express the Earth direction in terms of longitude
    #and latitude in the IAU_MOON frame.
    #
    [r, lon, lat] = spiceypy.reclat( imoonv )

    print( '\n'
           'Moon-Earth direction using low accuracy\n'
           'PCK and IAU_MOON frame:\n'
           'Earth lon (deg):        {0:15.6f}\n'
           'Earth lat (deg):        {1:15.6f}\n'.format(
               lon * spiceypy.dpr(),
               lat * spiceypy.dpr() )  )
    #
    # Look up the apparent position of the Earth relative
    # to the Moon's center in the MOON_ME frame at ET.
    #
    [mmoonv, ltime] = spiceypy.spkpos( 'earth', et, 'moon_me',
                                       'lt+s', 'moon'        )
    #
    # Express the Earth direction in terms of longitude
    # and latitude in the MOON_ME frame.
    #
    [r, lon, lat] = spiceypy.reclat( mmoonv )

    print( 'Moon-Earth direction using high accuracy\n'
           'PCK and MOON_ME frame:\n'
           'Earth lon (deg):        {0:15.6f}\n'
           'Earth lat (deg):        {1:15.6f}\n'.format(
               lon * spiceypy.dpr(),
               lat * spiceypy.dpr() )  )
    #
    # Find the angular separation of the Earth position
    # vectors in degrees.
    #
    sep = spiceypy.dpr() * spiceypy.vsep( imoonv, mmoonv )

    print( 'For IAU_MOON vs MOON_ME frames:' )
    print( 'Moon-Earth vector separation angle (deg):     '
           '{:15.6f}\n'.format( sep )  )
    #
    # Look up the apparent position of the Earth relative
    # to the Moon's center in the MOON_PA frame at ET.
    #
    [pmoonv, ltime] = spiceypy.spkpos( 'earth', et, 'moon_pa',
                                       'lt+s',  'moon'        )
    #
    # Express the Earth direction in terms of longitude
    # and latitude in the MOON_PA frame.
    #
    [r, lon, lat] = spiceypy.reclat( pmoonv )

    print( 'Moon-Earth direction using high accuracy\n'
           'PCK and MOON_PA frame:\n'
           'Earth lon (deg):        {0:15.6f}\n'
           'Earth lat (deg):        {1:15.6f}\n'.format(
               lon * spiceypy.dpr(),
               lat * spiceypy.dpr() )  )
    #
    # Find the angular separation of the Earth position
    # vectors in degrees.
    #
    sep = spiceypy.dpr() * spiceypy.vsep( pmoonv, mmoonv )

    print( 'For MOON_PA vs MOON_ME frames:' )
    print( 'Moon-Earth vector separation angle (deg):     '
           '{:15.6f}\n'.format( sep )  )
    #
    # Find the apparent sub-Earth point on the Moon at ET
    # using the MOON_ME frame.
    #
    [msub, trgepc, srfvec ] = spiceypy.subpnt(
        'near point: ellipsoid', 'moon',
        et,  'moon_me', 'lt+s',  'earth' )
    #
    # Display the sub-point in latitudinal coordinates.
    #
    [r, lon, lat] = spiceypy.reclat( msub )

    print( 'Sub-Earth point on Moon using high accuracy\n'
           'PCK and MOON_ME frame:\n'
           'Sub-Earth lon (deg):   {0:15.6f}\n'
           'Sub-Earth lat (deg):   {1:15.6f}\n'.format(
               lon * spiceypy.dpr(),
               lat * spiceypy.dpr()  )  )
    #
    # Find the apparent sub-Earth point on the Moon at
    # ET using the MOON_PA frame.
    #
    [psub, trgepc, srfvec] = spiceypy.subpnt(
        'near point: ellipsoid',  'moon',
         et,   'moon_pa', 'lt+s', 'earth'    )
    #
    # Display the sub-point in latitudinal coordinates.
    #
    [r, lon, lat] = spiceypy.reclat( psub )

    print( 'Sub-Earth point on Moon using high accuracy\n'
           'PCK and MOON_PA frame:\n'
           'Sub-Earth lon (deg):   {0:15.6f}\n'
           'Sub-Earth lat (deg):   {1:15.6f}\n'.format(
               lon * spiceypy.dpr(),
               lat * spiceypy.dpr() )  )
    #
    # Find the distance between the sub-Earth points
    # in km.
    #
    dist = spiceypy.vdist( msub, psub )

    print( 'Distance between sub-Earth points (km): '
           '{:15.6f}\n'.format( dist )  )

    spiceypy.unload( METAKR )

if __name__ == '__main__':
     mrotat()