# -*- coding: utf-8 -*-
"""
Material library data model. Separate from simulation/project.
Based on diaphragm_opencl default materials.
"""

from __future__ import annotations 

import json 
from dataclasses import asdict ,dataclass ,field 
from pathlib import Path 
from typing import Any 


@dataclass 
class MaterialEntry :
    """Single material: density..eta_visc, coupling_gain (pressure reception), acoustic_inject (air radiation source)."""

    name :str 
    density :float # kg/m³
    E_parallel :float # Pa
    E_perp :float # Pa
    poisson :float 
    Cd :float 
    eta_visc :float # Pa·s
    coupling_gain :float # reception from air-field, 0..1
    acoustic_inject :float =0.0 # contribution to acoustic injection (0 = no monopole into air grid)

    def to_row (self )->list [float ]:
        """Row for np.ndarray / diaphragm_opencl format (8 floats)."""
        return [
        self .density ,
        self .E_parallel ,
        self .E_perp ,
        self .poisson ,
        self .Cd ,
        self .eta_visc ,
        self .coupling_gain ,
        self .acoustic_inject ,
        ]

    @staticmethod 
    def from_row (name :str ,row :list [float ])->"MaterialEntry":
        r =list (row )+[0.0 ]*8 
        cg =float (r [6 ])
        nl =name .strip ().lower ()
        if len (row )>=8 :
            inj =float (r [7 ])
        elif nl =="sensor":
            inj =0.0 
        elif nl =="membrane":
            inj =1.0 
        else :
            inj =min (0.30 ,0.55 *cg )
        return MaterialEntry (
        name =name ,
        density =float (r [0 ]),
        E_parallel =float (r [1 ]),
        E_perp =float (r [2 ]),
        poisson =float (r [3 ]),
        Cd =float (r [4 ]),
        eta_visc =float (r [5 ]),
        coupling_gain =cg ,
        acoustic_inject =inj ,
        )

    def to_dict (self )->dict [str ,Any ]:
        return asdict (self )

    @staticmethod 
    def from_dict (d :dict [str ,Any ])->"MaterialEntry":
        cg =float (d .get ("coupling_gain",d .get ("coupling_recv",0.5 )))
        inj =d .get ("acoustic_inject")
        name_l =str (d .get ("name","")).strip ().lower ()
        if inj is None :
            if name_l =="sensor":
                inj =1.0 
            elif name_l =="membrane":
                inj =1.0 
            else :
                inj =1.0 
        else :
            inj =float (inj )
        return MaterialEntry (
        name =str (d .get ("name","Unnamed")),
        density =float (d .get ("density",1000.0 )),
        E_parallel =float (d .get ("E_parallel",1e9 )),
        E_perp =float (d .get ("E_perp",1e9 )),
        poisson =float (d .get ("poisson",0.3 )),
        Cd =float (d .get ("Cd",1.0 )),
        eta_visc =float (d .get ("eta_visc",1.0 )),
        coupling_gain =cg ,
        acoustic_inject =inj ,
        )


def _stock_materials ()->list [MaterialEntry ]:
    """Default library from diaphragm_opencl._build_default_material_library."""
    return [
    MaterialEntry (
    name ="membrane",
    density =1380.0 ,
    E_parallel =5.0e9 ,
    E_perp =3.5e9 ,
    poisson =0.30 ,
    Cd =1.0 ,
    eta_visc =0.8 ,
    coupling_gain =0.90 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="foam_ve3015",
    density =350.0 ,
    E_parallel =0.08e6 ,
    E_perp =0.05e6 ,
    poisson =0.30 ,
    Cd =1.20 ,
    eta_visc =150.0 ,
    coupling_gain =0.25 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="sheepskin_leather",
    density =998.0 ,
    E_parallel =10.0e6 ,
    E_perp =7.0e6 ,
    poisson =0.40 ,
    Cd =1.05 ,
    eta_visc =12.0 ,
    coupling_gain =0.60 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="human_ear_avg",
    density =1080.0 ,
    E_parallel =1.80e6 ,
    E_perp =1.50e6 ,
    poisson =0.45 ,
    Cd =1.10 ,
    eta_visc =20.0 ,
    coupling_gain =0.50 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="sensor",
    density =1380.0 ,
    E_parallel =5.0e9 ,
    E_perp =3.5e9 ,
    poisson =0.30 ,
    Cd =1.0 ,
    eta_visc =0.8 ,
    coupling_gain =1.00 ,
    acoustic_inject =0.0 ,
    ),
    MaterialEntry (
    name ="cotton_wool",
    density =250.0 ,
    E_parallel =0.03e6 ,
    E_perp =0.02e6 ,
    poisson =0.20 ,
    Cd =1.35 ,
    eta_visc =220.0 ,
    coupling_gain =0.30 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="abs_plastic",
    density =1050.0 ,
    E_parallel =2.4e9 ,
    E_perp =2.4e9 ,
    poisson =0.36 ,
    Cd =1.05 ,
    eta_visc =15.0 ,
    coupling_gain =0.55 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="neodymium_magnet",
    density =7500.0 ,
    E_parallel =160.0e9 ,
    E_perp =160.0e9 ,
    poisson =0.24 ,
    Cd =1.0 ,
    eta_visc =1.0 ,
    coupling_gain =0.15 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="stainless_steel_303",
    density =8000.0 ,
    E_parallel =195.0e9 ,
    E_perp =195.0e9 ,
    poisson =0.27 ,
    Cd =1.0 ,
    eta_visc =1.0 ,
    coupling_gain =0.12 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="silicone_30_shore_a",
    density =1100.0 ,
    E_parallel =2.5e6 ,
    E_perp =2.5e6 ,
    poisson =0.49 ,
    Cd =1.08 ,
    eta_visc =8.0 ,
    coupling_gain =0.65 ,
    acoustic_inject =1.0 ,
    ),
    MaterialEntry (
    name ="air",
    density =1.225 ,
    E_parallel =1.42e5 ,
    E_perp =1.42e5 ,
    poisson =0.0 ,
    Cd =0.0 ,
    eta_visc =1.8e-5 ,
    coupling_gain =1.00 ,
    acoustic_inject =0.0 ,
    ),
    ]


class MaterialLibraryModel :
    """Library of materials. Separate data model, not tied to project."""

    def __init__ (self )->None :
        self ._materials :list [MaterialEntry ]=list (_stock_materials ())

    @property 
    def materials (self )->list [MaterialEntry ]:
        return list (self ._materials )

    def add (self ,entry :MaterialEntry )->int :
        self ._materials .append (entry )
        return len (self ._materials )-1 

    def update (self ,index :int ,entry :MaterialEntry )->None :
        if 0 <=index <len (self ._materials ):
            self ._materials [index ]=entry 

    def remove (self ,index :int )->None :
        if 0 <=index <len (self ._materials ):
            self ._materials .pop (index )

    def clear_and_reset_to_stock (self )->None :
        """Reset library to stock materials from diaphragm_opencl."""
        self ._materials =list (_stock_materials ())

    def merge (self ,entries :list [MaterialEntry ])->None :
        """Merge imported entries. Adds new materials, skips duplicates by name."""
        existing_names ={m .name .lower ()for m in self ._materials }
        for e in entries :
            if e .name .lower ()not in existing_names :
                self ._materials .append (e )
                existing_names .add (e .name .lower ())

    def _unique_name (self ,base :str )->str :
        'Returns a unique name: base, base2, base3, ... if there are conflicts.'
        existing ={m .name .lower ()for m in self ._materials }
        name =base 
        n =1 
        while name .lower ()in existing :
            n +=1 
            name =f"{base }{n }"
        return name 

    def ensure_material (
    self ,
    name :str ,
    density :float ,
    E_parallel :float ,
    E_perp :float ,
    poisson :float ,
    Cd :float ,
    eta_visc :float ,
    coupling_gain :float ,
    acoustic_inject :float =0.0 ,
    )->str :
        'Ensures the availability of material in the library. If name already exists, it returns name.\n        If not, adds it with the specified properties. If there is a name conflict, the number is added.\n        Returns the actual material name (to update mesh.material_key).'
        for m in self ._materials :
            if m .name .lower ()==name .lower ():
                return m .name 
        unique =self ._unique_name (name )
        entry =MaterialEntry (
        name =unique ,
        density =density ,
        E_parallel =E_parallel ,
        E_perp =E_perp ,
        poisson =poisson ,
        Cd =Cd ,
        eta_visc =eta_visc ,
        coupling_gain =coupling_gain ,
        acoustic_inject =acoustic_inject ,
        )
        self ._materials .append (entry )
        return unique 

    def to_numpy_array (self )->"np.ndarray":
        """Export as np.ndarray for diaphragm_opencl.set_material_library."""
        import numpy as np 
        return np .array ([m .to_row ()for m in self ._materials ],dtype =np .float64 )

    def save_json (self ,path :str |Path )->None :
        data ={"materials":[m .to_dict ()for m in self ._materials ]}
        Path (path ).write_text (json .dumps (data ,ensure_ascii =False ,indent =2 ),encoding ="utf-8")

    def load_json (self ,path :str |Path )->list [MaterialEntry ]:
        data =json .loads (Path (path ).read_text (encoding ="utf-8"))
        mats =data .get ("materials",[])
        return [MaterialEntry .from_dict (m )for m in mats ]

    def load_from_file (self ,path :str |Path )->None :
        """Replace library with contents of JSON file."""
        self ._materials =self .load_json (path )

    def import_and_merge (self ,path :str |Path )->int :
        """Import from JSON and merge. Returns count of added materials."""
        imported =self .load_json (path )
        before =len (self ._materials )
        self .merge (imported )
        return len (self ._materials )-before 

    def export_json (self ,path :str |Path )->None :
        self .save_json (path )
