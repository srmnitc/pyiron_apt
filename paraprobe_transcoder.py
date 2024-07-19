#
# This file is part of paraprobe-toolbox.
#
# paraprobe-toolbox is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License,
#  or (at your option) any later version.
#
# paraprobe-toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with paraprobe-toolbox. If not, see <https://www.gnu.org/licenses/>.
#
"""Transcode atom probe file formats into NeXus/HDF5 for paraprobe-toolbox."""

import os
import time
import datetime as dt
import numpy as np
import h5py

from ifes_apt_tc_data_modeling.apt.apt6_reader import ReadAptFileFormat
from ifes_apt_tc_data_modeling.ato.ato_reader import ReadAtoFileFormat
from ifes_apt_tc_data_modeling.csv.csv_reader import ReadCsvFileFormat
from ifes_apt_tc_data_modeling.env.env_reader import ReadEnvFileFormat
from ifes_apt_tc_data_modeling.epos.epos_reader import ReadEposFileFormat
from ifes_apt_tc_data_modeling.fig.fig_reader import ReadFigTxtFileFormat
from ifes_apt_tc_data_modeling.imago.imago_reader import ReadImagoAnalysisFileFormat
from ifes_apt_tc_data_modeling.pos.pos_reader import ReadPosFileFormat
from ifes_apt_tc_data_modeling.pyccapt.pyccapt_reader \
    import ReadPyccaptCalibrationFileFormat, ReadPyccaptRangingFileFormat
from ifes_apt_tc_data_modeling.rng.rng_reader import ReadRngFileFormat
from ifes_apt_tc_data_modeling.rrng.rrng_reader import ReadRrngFileFormat

from ifes_apt_tc_data_modeling.utils.utils import MAX_NUMBER_OF_ATOMS_PER_ION
from paraprobe_utils.gitinfo import PARAPROBE_VERSION, NEXUS_VERSION
from paraprobe_utils.hashing import get_file_hashvalue
from paraprobe_utils.numerics import MYHDF5_COMPRESSION_DEFAULT
from paraprobe_utils.hashing import get_simulation_id

# in the long run the paraprobe-transcoder tool should be obsolete as
# www.github.com/FAIRmat-NFDI/pynxtools/pynxtools/dataconverter/readers/apm
# generates a compliant NeXus file


class ParaprobeTranscoder():
    """The paraprobe-transcoder tool."""

    def __init__(self, nxs_config_file_name):
        """Initialize the tool with configurations."""
        self.config_file = nxs_config_file_name
        self.entry_id = 1
        self.task_id = 1
        self.simid = get_simulation_id(nxs_config_file_name)
        self.start_time = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        self.end_time = None

        with h5py.File(self.config_file, "r") as h5r:
            # are we facing a directly applicable NXapm-compliant NeXus/HDF5 file or not
            path_recon = f"/entry{self.entry_id}/transcode/reconstruction/path"
            path_range = f"/entry{self.entry_id}/transcode/ranging/path"
            print(f"{path_recon}, {path_range}")
            self.transcoding_needed = (not h5r[path_recon][()].decode("utf8").endswith(".nxs")) \
                or (not h5r[path_range][()].decode("utf8").endswith(".nxs")) \
                or (h5r[path_recon][()].decode("utf8") != h5r[path_range][()].decode("utf8"))
            print(self.transcoding_needed)
            self.reconstructed_dataset = h5r[path_recon][()].decode("utf8")
            self.ranging_definitions = h5r[path_range][()].decode("utf8")
            self.resultsfile = f"PARAPROBE.Transcoder.Results.SimID.{self.simid}.nxs"
            print(f"Processing configuration file: {self.config_file}")
            print(f"Processing reconstruction: {self.reconstructed_dataset}")
            print(f"Processing ranging: {self.ranging_definitions}")
            print(f"Results file: {self.resultsfile}")
            if self.transcoding_needed is True:
                print("Input reconstruction and ranging definitions use files from")
                print("technology partners (POS, ePOS, APT, RRNG, RNG) or other")
                print("file formats from the community. These will be transcoded to NeXus...")
            else:
                print("Input reconstruction and ranging definitions use NeXus already.")
                print("Hence, only xdmf_topology array for visualization will be added.")

    def load_apt_six(self, h5_handle):
        """Transcode data from APT(6) to NeXus/HDF5."""
        apt_reader = ReadAptFileFormat(self.reconstructed_dataset)

        xyz = apt_reader.get_reconstructed_positions()
        number_of_ions = np.shape(xyz.values)[0]
        dset_name = f"/entry{self.entry_id}/atom_probe/reconstruction/reconstructed_positions"
        dst = h5_handle.create_dataset(dset_name,
                                       dtype=np.float32,
                                       data=xyz.values[:, :])
        dst.attrs["unit"] = xyz.unit
        del xyz

        m_z = apt_reader.get_mass_to_charge_state_ratio()
        dset_name = f"/entry{self.entry_id}/atom_probe/mass_to_charge_conversion/mass_to_charge"
        dst = h5_handle.create_dataset(dset_name,
                                       dtype=np.float32,
                                       data=m_z.values[:, 0])
        dst.attrs["unit"] = m_z.unit
        del m_z

        return number_of_ions

    def load_ato(self, h5_handle):
        return 0

    def load_csv(self, h5_handle):
        return 0

    def load_imago(self, h5_handle):
        pass

    def load_epos(self, h5_handle):
        """Transcode data from ePOS to NeXus/HDF5."""
        epos_reader = ReadEposFileFormat(self.reconstructed_dataset)

        xyz = epos_reader.get_reconstructed_positions()
        number_of_ions = np.shape(xyz.values)[0]
        dset_name = f"/entry{self.entry_id}/atom_probe/reconstruction/reconstructed_positions"
        dst = h5_handle.create_dataset(dset_name,
                                       dtype=np.float32,
                                       data=xyz.values)
        dst.attrs["unit"] = xyz.unit
        del xyz

        m_z = epos_reader.get_mass_to_charge_state_ratio()
        dset_name = f"/entry{self.entry_id}/atom_probe/mass_to_charge_conversion/mass_to_charge"
        dst = h5_handle.create_dataset(dset_name,
                                       dtype=np.float32,
                                       data=m_z.values)
        dst.attrs["unit"] = m_z.unit
        del m_z

        return number_of_ions

    def load_pos(self, h5_handle):
        """Transcode data from POS to NeXus/HDF5."""
        pos_reader = ReadPosFileFormat(self.reconstructed_dataset)

        xyz = pos_reader.get_reconstructed_positions()
        number_of_ions = np.shape(xyz.values)[0]
        dset_name = f"/entry{self.entry_id}/atom_probe/reconstruction/reconstructed_positions"
        dst = h5_handle.create_dataset(dset_name,
                                       dtype=np.float32,
                                       data=xyz.values)
        dst.attrs["unit"] = xyz.unit
        del xyz

        m_z = pos_reader.get_mass_to_charge_state_ratio()
        dset_name = f"/entry{self.entry_id}/atom_probe/mass_to_charge_conversion/mass_to_charge"
        dst = h5_handle.create_dataset(dset_name,
                                       dtype=np.float32,
                                       data=m_z.values)
        dst.attrs["unit"] = m_z.unit
        del m_z

        return number_of_ions

    def load_pyccapt_calibration(self, h5_handle):
        return 0

    def load_env(self, h5_handle):
        pass

    def load_figtxt(self, h5_handle):
        """Transcode ranging definitions from Matlab FIG file."""
        """
        Transcode ranging data from Matlab figures to NeXus/HDF5.
        this parser is meant to serve P. Felfer"s group at Erlangen-Nuernberg
        they store ranging definitions via their so-called atom probe toolbox
        This Matlab tool stores all ranges in a Matlab figure, i.e. they
        write essentially Matlab -v7.3 figures. These are HDF5 under the hood
        but use proprietary customizations that Matlab implemented internally
        in the figure management and I/O subsystem, this is a barrier for
        FAIR atom probe research because this effectively makes these fig
        files no longer safely interpretable with Python.
        As a workaround for now a conversion step is looped into the workflow
        Specifically, paraprobe-parmsetup/src/cxx/matlab comes with a
        matlab utility script which can be used to load the figure in matlab
        and write a HDF5 utility file with the relevant ranging data in the
        figure. This function here parses then for now from this intermediate
        format into the default format used by the paraprobe-toolbox.
        If the Erlangen group would write this HDF5 file as a supplementary
        step into their atom probe toolbox, their tools would be easier
        interoperable with the paraprobe toolbox
        """
        pass
        # fig_reader = ReadMatlabFigFileFormat(self.ranging_definitions)
        # fig_reader.read()
        # self.load_ranging(h5_handle, fig_reader.fig["ions"], "fig")
        # print(f"Transcoded ranging data for {self.ranging_definitions}")

    def load_pyccapt_ranging(self, h5_handle):
        pass

    def load_rng(self, h5_handle):
        """Transcode data from RNG to NeXus/HDF5."""
        rng_reader = ReadRngFileFormat(self.ranging_definitions)
        self.load_ranging(h5_handle, rng_reader.rng["molecular_ions"])
        print(f"Transcoded ranging data for {self.ranging_definitions}")

    def load_rrng(self, h5_handle):
        """Transcode data from RRNG to NeXus/HDF5."""
        rrng_reader = ReadRrngFileFormat(self.ranging_definitions)
        self.load_ranging(h5_handle, rrng_reader.rrng["molecular_ions"])
        print(f"Transcoded ranging data for {self.ranging_definitions}")

    def add_unknown_iontype(self, prfx, h5_handle):
        """Add unknown special iontype used for all ions that have no range."""
        iontype_id = 0
        trg = f"{prfx}/ion{iontype_id}"
        grp = h5_handle.create_group(f"{trg}")
        grp.attrs["NX_class"] = "NXion"
        dst = h5_handle.create_dataset(f"{trg}/charge_state",
                                       data=np.int8(0))
        dst.attrs["comment"] = "charge_state of the special UNKNOWN_IONTYPE is always 0."
        dst = h5_handle.create_dataset(f"{trg}/nuclide_hash",
                                       data=np.zeros((MAX_NUMBER_OF_ATOMS_PER_ION,), np.uint16),
                                       chunks=True,
                                       compression="gzip",
                                       compression_opts=MYHDF5_COMPRESSION_DEFAULT)
        dst.attrs["comment"] \
            = "Hashing rule is that each np.uint16 is a hashval = n_protons + 256 * n_neutrons"
        dst = h5_handle.create_dataset(f"{trg}/nuclide_list",
                                       data=np.zeros((MAX_NUMBER_OF_ATOMS_PER_ION, 2), np.uint16),
                                       chunks=True,
                                       compression="gzip",
                                       compression_opts=MYHDF5_COMPRESSION_DEFAULT)
        dst = h5_handle.create_dataset(f"{trg}/mass_to_charge_range",
                                       shape=(1, 2),
                                       data=np.asarray([0.000, 0.001], np.float32))
        dst.attrs["unit"] = "Da"
        dst = h5_handle.create_dataset(
            f"{trg}/name", data="unknown iontype")
        # (1,), dtype="S"+str(len(ion_name))), dst[0] = np.string_(ion_name)

    def load_ranging(self, h5_handle, molecular_ion_dict):
        """Transcode ranging metadata to NeXus/HDF file."""
        grp_name = f"/entry{self.entry_id}/atom_probe/ranging/peak_identification"
        grp = h5_handle.create_group(grp_name)
        grp.attrs["NX_class"] = "NXprocess"

        self.add_unknown_iontype(grp_name, h5_handle)

        iontype_id = 1
        for ion in molecular_ion_dict:
            sub_grp_name = f"{grp_name}/ion{iontype_id}"
            grp = h5_handle.create_group(sub_grp_name)
            grp.attrs["NX_class"] = "NXion"

            dst = h5_handle.create_dataset(f"{sub_grp_name}/charge_state",
                                           data=np.int8(ion.charge_state.values))
            if len(ion.name.values) > 0:
                dst = h5_handle.create_dataset(f"{sub_grp_name}/name",
                                               data=ion.name.values)
            dst = h5_handle.create_dataset(f"{sub_grp_name}/nuclide_hash",
                                           data=ion.nuclide_hash.values,
                                           chunks=True,
                                           compression="gzip",
                                           compression_opts=MYHDF5_COMPRESSION_DEFAULT)
            dst = h5_handle.create_dataset(f"{sub_grp_name}/nuclide_list",
                                           data=ion.nuclide_list.values,
                                           chunks=True,
                                           compression="gzip",
                                           compression_opts=MYHDF5_COMPRESSION_DEFAULT)
            dset_name = f"{sub_grp_name}/mass_to_charge_range"
            dst = h5_handle.create_dataset(dset_name,
                                           shape=np.shape(ion.ranges.values),
                                           dtype=np.float32,
                                           data=ion.ranges.values)
            dst.attrs["unit"] = "Da"
            iontype_id += 1

    def query_nxs_reconstruction(self):
        """Identify how many ions in the dataset."""
        number_of_ions = 0
        with h5py.File(self.reconstructed_dataset, "r") as h5r:
            dset_name = f"/entry{self.entry_id}/atom_probe/reconstruction/reconstructed_positions"
            number_of_ions = np.shape(h5r[dset_name][:, :])[0]
        return number_of_ions

    def execute(self):
        """Create a NeXuS/HDF5-conformant input file for paraprobe."""
        tic = time.perf_counter()
        # there are two different input file scenarios:
        # tech partner files: i. e. APT, ATO, ENV, FIG, POS, ePOS, RNG, RRNG or
        # NeXus NXapm-compliant HDF5 files: i. e. NXS
        with h5py.File(self.resultsfile, "w") as h5w:
            trg = f"/entry{self.entry_id}"
            grp = h5w.create_group(f"{trg}")
            grp.attrs["NX_class"] = "NXentry"
            dst = h5w.create_dataset(f"{trg}/definition",
                                     data="NXapm_paraprobe_transcoder_results")
            dst.attrs["version"] = NEXUS_VERSION
            grp = h5w.create_group(f"{trg}/common")
            grp.attrs["NX_class"] = "NXapm_paraprobe_tool_common"

            trg = f"/entry{self.entry_id}/common"
            dst = h5w.create_dataset(f"{trg}/analysis_identifier", data=self.simid)
            trg = f"/entry{self.entry_id}/common/config"
            grp = h5w.create_group(f"{trg}")
            grp.attrs["NX_class"] = "NXserialized"
            dst = h5w.create_dataset(f"{trg}/type", data="file")
            dst = h5w.create_dataset(f"{trg}/path", data=self.config_file)
            dst = h5w.create_dataset(f"{trg}/checksum", data=get_file_hashvalue(self.config_file))
            dst = h5w.create_dataset(f"{trg}/algorithm", data="sha256")

            trg = f"/entry{self.entry_id}/common/program1"
            grp = h5w.create_group(f"{trg}")
            grp.attrs["NX_class"] = "NXprogram"
            dst = h5w.create_dataset(f"{trg}/program",
                                     data="paraprobe-toolbox-transcoder")
            dst.attrs["version"] = PARAPROBE_VERSION

            trg = f"/entry{self.entry_id}/common/profiling"
            grp = h5w.create_group(f"{trg}")
            grp.attrs["NX_class"] = "NXcs_profiling"
            dst = h5w.create_dataset(f"{trg}/start_time",
                                     data=self.start_time)

            trg = f"/entry{self.entry_id}/atom_probe"
            trg = f"/entry{self.entry_id}/atom_probe"
            grp = h5w.create_group(f"{trg}")
            grp.attrs["NX_class"] = "NXapm_paraprobe_tool_results"
            for req in ["reconstruction", "mass_to_charge_conversion", "ranging"]:
                grp = h5w.create_group(f"{trg}/{req}")
                grp.attrs["NX_class"] = "NXprocess"

            number_of_ions = 0
            if self.transcoding_needed is True:
                ###
                if self.reconstructed_dataset.lower().endswith(".apt"):
                    number_of_ions = self.load_apt_six(h5w)
                elif self.reconstructed_dataset.lower().endswith(".ato"):
                    number_of_ions = self.load_ato(h5w)
                elif self.reconstructed_dataset.lower().endswith(".csv"):
                    number_of_ions = self.load_csv(h5w)
                elif self.reconstructed_dataset.lower().endswith(".epos"):
                    number_of_ions = self.load_epos(h5w)
                elif self.reconstructed_dataset.lower().endswith(".pos"):
                    number_of_ions = self.load_pos(h5w)
                elif self.reconstructed_dataset.lower().endswith(".h5"):
                    number_of_ions = self.load_pyccapt_calibration(h5w)
                else:
                    print("Supported file formats are *.apt, *.ato, *.csv, *.epos, *.pos, or *.h5 (pyccapt) !")
                print("Transcoded reconstruction and mass-to-charge-state-ratio values")

                if self.ranging_definitions.lower().endswith(".analysis"):
                    self.load_imago(h5w)
                elif self.ranging_definitions.lower().endswith(".env"):
                    self.load_env(h5w)
                elif self.ranging_definitions.lower().endswith(".h5"):
                    self.load_pyccapt_ranging(h5w)
                elif self.ranging_definitions.lower().endswith(".fig.txt"):
                    self.load_figtxt(h5w)
                elif self.ranging_definitions.lower().endswith(".rng"):
                    self.load_rng(h5w)
                if self.ranging_definitions.lower().endswith(".rrng"):
                    self.load_rrng(h5w)
                else:
                    print("Supported file formats are *.analysis, *.env, *.h5, *.fig.txt, *.rng, *.rrng !")
                print("Transcoded ranging definitions")
            else:
                # extract number_of_ions from NeXus File
                number_of_ions = self.query_nxs_reconstruction()
                print(f"Skipping transcoding because using NeXus is ready")

            # completed handling of either community or NXapm input files

            trg = f"/entry{self.entry_id}/common/coordinate_system_set"
            grp = h5w.create_group(f"{trg}")
            grp.attrs["NX_class"] = "NXcoordinate_system_set"
            grp = h5w.create_group(f"{trg}/paraprobe")
            grp.attrs["NX_class"] = "NXcoordinate_system"
            # dst = h5w.create_dataset(f"{trg}/paraprobe/offset", data="[0., 0., 0.]")
            dst = h5w.create_dataset(f"{trg}/paraprobe/type", data="cartesian")
            dst = h5w.create_dataset(f"{trg}/paraprobe/handedness", data="right_handed")
            cs_xyz = np.asarray([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]], np.float32)  # x, y, z
            for idx, axis_name in enumerate(["x", "y", "z"]):
                dst = h5w.create_dataset(f"{trg}/paraprobe/{axis_name}",
                                         data=cs_xyz[:, idx])
                # dst.attrs["offset"] = np.asarray([0., 0., 0.], np.float32)
                # dst.attrs["offset_units"] = "nm"
                # dst.attrs["depends_on"] = "."

            # add topology field
            if number_of_ions == 0:
                raise ValueError("Number of ions should be larger than 0 !")
            topology = np.ones([3 * number_of_ions], np.uint32)
            # triplets: xdmf polyvertex, xdmf vertex counter, vertex id
            # number_of_ions many such triplets
            topology[2:3 * number_of_ions:3] \
                = np.uint32(np.linspace(0, number_of_ions - 1, num=number_of_ions))

            trg = f"/entry{self.entry_id}/atom_probe/reconstruction"
            grp = h5w.create_group(f"{trg}/visualization")
            grp.attrs["NX_class"] = "NXprocess"
            dst = h5w.create_dataset(f"{trg}/visualization/xdmf_topology",
                                     data=topology,
                                     chunks=True,
                                     compression="gzip",
                                     compression_opts=MYHDF5_COMPRESSION_DEFAULT)
            dst.attrs["comments"] \
                = "An array of triplets. As many triplets as number of ions. " \
                  "First value is the XDMF primitive type keyword (here 1 for polyvertex). " \
                  "Second value is the number of primitives (here 1 because each point rendered). " \
                  "Third value is the ion identifier (here starting at 0)."
            del topology

            # datetime.datetime.now().astimezone().isoformat()
            trg = f"/entry{self.entry_id}/common/profiling"
            dst = h5w.create_dataset(f"{trg}/current_working_directory",
                                     data=os.getcwd())
            for opt_name, opt_value in [("processes", 1), ("threads", 1), ("gpus", 0)]:
                dst = h5w.create_dataset(f"{trg}/number_of_{opt_name}",
                                         data=np.uint32(opt_value))
            dst = h5w.create_dataset(f"/entry{self.entry_id}/common/status", data="success")
            self.end_time = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            dst = h5w.create_dataset(f"{trg}/end_time",
                                     data=self.end_time)
            toc = time.perf_counter()
            dst = h5w.create_dataset(f"{trg}/total_elapsed_time", data=toc - tic)
            dst.attrs["unit"] = "s"
        print("paraprobe-transcoder success")
        return self.resultsfile
