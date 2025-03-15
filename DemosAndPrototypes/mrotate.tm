KPL/MK

Meta-kernel for the "Moon Rotation" task in the Binary PCK
Hands On Lesson.

The names and contents of the kernels referenced by this
meta-kernel are as follows:

File name                    Contents
---------------------------  ------------------------------------
naif0008.tls                 Generic LSK
de414_2000_2020.bsp          Solar System Ephemeris
moon_060721.tf               Lunar FK
pck00008.tpc                 NAIF text PCK
moon_pa_de403_1950-2198.bpc  Moon binary PCK

\begindata

   KERNELS_TO_LOAD = ( 'Y:/kernels/lsk/naif0012.tls'
                       'Y:/kernels/spk/planets/de440.bsp'
                       'Y:/kernels/fk/satellites/moon_080317.tf'
                       'Y:/kernels/pck/pck00010.tpc'
                       'Y:/kernels/pck/moon_pa_de421_1900-2050.bpc' )
\begintext