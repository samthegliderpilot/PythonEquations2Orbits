import czml3 as czml
import czml3.types as czmlTypes
import czml3.properties as czmlProp
import czml3.enums as czmlEnums
import pyeq2orb.Graphics.Primitives as prim
from typing import List
from datetime import datetime, timedelta

def createCzmlFromPoints(epoch : datetime, name : str, points : List[prim.PathPrimitive]) ->czml.Document :

    earliest = points[0].ephemeris.T[0]
    latest = points[0].ephemeris.T[-1]

    point0CartesianNumbers = []

    for i in range(0, len(points[0].ephemeris.T)):
        point0CartesianNumbers.append(points[0].ephemeris.T[i])
        point0CartesianNumbers.append(points[0].ephemeris.X[i])
        point0CartesianNumbers.append(points[0].ephemeris.Y[i])
        point0CartesianNumbers.append(points[0].ephemeris.Z[i])


    for i in range(1, len(points)) :
        if points[i].ephemeris.T[0] < earliest :
            earliest = points[i].ephemeris.T[0]
        if points[i].ephemeris.T[-1] > latest :
            latest = points[i].ephemeris.T[-1]

    step = (latest-earliest) / 1000
    theDocument = czml.Document([
        czml.Preamble(
        name=name,
        clock = czmlTypes.IntervalValue(start=epoch, end=epoch + timedelta(seconds=latest), value=czmlProp.Clock(currentTime=epoch, multiplier=step)),
        ),
        czml.Packet(
            id = points[0].id,
            name = points[0].id,
            billboard = czmlProp.Billboard(horizontalOrigin=czmlProp.HorizontalOrigins.CENTER,
                    image=(
                        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9"
                        "hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdv"
                        "qGQAAADJSURBVDhPnZHRDcMgEEMZjVEYpaNklIzSEfLfD4qNnXAJSFWfhO7w2Zc0T"
                        "f9QG2rXrEzSUeZLOGm47WoH95x3Hl3jEgilvDgsOQUTqsNl68ezEwn1vae6lceSEE"
                        "YvvWNT/Rxc4CXQNGadho1NXoJ+9iaqc2xi2xbt23PJCDIB6TQjOC6Bho/sDy3fBQT"
                        "8PrVhibU7yBFcEPaRxOoeTwbwByCOYf9VGp1BYI1BA+EeHhmfzKbBoJEQwn1yzUZt"
                        "yspIQUha85MpkNIXB7GizqDEECsAAAAASUVORK5CYII="
                    ),
                    scale=1.5,
                    show=True,
                    verticalOrigin=czmlProp.VerticalOrigins.CENTER,),
        
            label=czmlProp.Label(
                horizontalOrigin=czmlProp.HorizontalOrigins.LEFT,
                outlineWidth=2,
                show=True,
                font="11pt Lucida Console",
                style=czmlProp.LabelStyles.FILL_AND_OUTLINE,
                text=points[0].id,
                verticalOrigin=czmlProp.VerticalOrigins.CENTER,
                fillColor=czmlProp.Color.from_list([0, 255, 0]),
                outlineColor=czmlProp.Color.from_list([0, 0, 0]),
            ),
            path=czmlProp.Path(
                show=czmlTypes.Sequence([czmlTypes.IntervalValue(start=epoch, end=epoch + timedelta(seconds=latest), value=True)]),
                width=2,
                resolution=2,
                material=czmlProp.Material(solidColor=czmlProp.SolidColorMaterial.from_list([0, 255, 0])),
                leadTime=8.64e5,
                trailTime=8.64e5,
            ),            
            position=czmlProp.Position(
                interpolationAlgorithm=czmlEnums.InterpolationAlgorithms.LAGRANGE,
                interpolationDegree=5,
                referenceFrame=czmlEnums.ReferenceFrames.INERTIAL,
                epoch=epoch,

                cartesian=point0CartesianNumbers,
                ),
        ),
    ])

    return theDocument


