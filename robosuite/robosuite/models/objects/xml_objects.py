from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion
import numpy as np

class ThreeDNet(MujocoXMLObject):
    """
    3DNet object
    """

    def __init__(self, N):
        #N = 100
        #random_n = np.random.randint(N)
        N = N % 1367
        #print(N)
        xml_path = "objects/3dnet_" + str(N) + ".xml"
        #xml_path = "objects/cup_" + str(N) + ".xml"
        super().__init__(xml_path_completion(xml_path))

class CustomObject(MujocoXMLObject):
    """
    custom object
    """

    def __init__(self, name):
        xml_path = "objects/" + str(name) + ".xml"
        super().__init__(xml_path_completion(xml_path))

class CustomRandomizedObject(MujocoXMLObject):
    """
    custom object
    """

    def __init__(self, name):#, r, g, b, a):
        xml_path = "objects/" + str(name) + ".xml"
        super().__init__(xml_path_completion(xml_path))

        """meshes = self.asset.findall("mesh")
        for i in range(len(meshes)):
            meshes[i].set("scale", "1.0 1.0 1.0")

        geoms = self.worldbody.findall("./body/body/geom")
        for i in range(len(geoms)):
            geoms[i].set("rgba", str(r) + " " + str(g) + " " + str(b) + " " + str(a))"""

        '''collision = copy.deepcopy(self.worldbody.find("./body/body[@name='collision']"))
        collision.attrib.pop("name")
        if name is not None:
            collision.attrib["name"] = name
            geoms = collision.findall("geom")
            if len(geoms) == 1:
                geoms[0].set("name", name)
            else:
                for i in range(len(geoms)):
                    geoms[i].set("name", "{}-{}".format(name, i))
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            collision.append(ET.Element("site", attrib=template))
        return collision'''

class Rusk(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/Rusk.xml"))

class InstantSoup(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/InstantSoup.xml"))


class CokePlasticSmallGrasp(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/CokePlasticSmallGrasp.xml"))

class LivioClassicOil(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/LivioClassicOil.xml"))

class Shampoo(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/Shampoo.xml"))

class ShowerGel(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/ShowerGel.xml"))

class Sprayflask(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/Sprayflask.xml"))

class Toothpaste(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/Toothpaste.xml"))

class BlueSaltCube(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/BlueSaltCube.xml"))

class CoffeeBox(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/CoffeeBox.xml"))
class FlowerCup(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/FlowerCup.xml"))
class Glassbowl(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/Glassbowl.xml"))
class GreenCup(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/GreenCup.xml"))
class GreenSaltCylinder(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/GreenSaltCylinder.xml"))
class RedCup(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/RedCup.xml"))
class SmallGlass(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/SmallGlass.xml"))
class Waterglass(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/Waterglass.xml"))
class YellowSaltCube(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/YellowSaltCube.xml"))


class obj0(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/obj0.xml"))

class obj1(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/obj1.xml"))

class obj2(MujocoXMLObject):
    """
    obj0 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/obj2.xml"))

class MugObject(MujocoXMLObject):
    """
    Mug object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/mug.xml"))

class L2Object(MujocoXMLObject):
    """
    L2 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/L2.xml"))

class LObject(MujocoXMLObject):
    """
    L object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/L.xml"))

class TObject(MujocoXMLObject):
    """
    T object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/T.xml"))

class TlargeObject(MujocoXMLObject):
    """
    T object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/T_large.xml"))

class roundTlargeObject(MujocoXMLObject):
    """
    T object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/round_T_large.xml"))

class UObject(MujocoXMLObject):
    """
    U object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/U.xml"))

class Bottle1Object(MujocoXMLObject):
    """
    Bottle1 object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bottle1.xml"))

class ConcaveObject(MujocoXMLObject):
    """
    Concave object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/concaveobj.xml"))


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bottle.xml"))


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/can.xml"))


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/lemon.xml"))


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/milk.xml"))


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bread.xml"))


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cereal.xml"))


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/square-nut.xml"))


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/round-nut.xml"))


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in SawyerPickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/milk-visual.xml"))


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bread-visual.xml"))


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cereal-visual.xml"))


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/can-visual.xml"))


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in BaxterPegInHole)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/plate-with-hole.xml"))
