"""
Master class for fusing aerial IR analysis with time series data.
"""

from shapely.geometry.polygon import Polygon
import pandas as pd
import plotly.express as px
import folium
import matplotlib.pyplot as plt


class DataFusion:

    def __init__(self, defect_category_dict, by_site=False, by_block=False):
        self.defect_category_dict = defect_category_dict
        self.by_site = by_site
        self.by_block = by_block

    def merge_aerial_site_dictionary(self, aerial_defect_dict, site_dict,
                                     defect_mapping_dict, metadata):
        """
        Merge a dictionary containing aerial defect information with a site
        location dictionary. Returns a dataframe where each row is a
        defect found, mapping to a particular area in the site dictionary.
        Parameter:
            aerial_defect_dict: Dictionary
                Dictionary from a geojson file that contains the defect 
                location
            site_dict: Dictionary 
                Dictionary from a geojson file that contains the site and all 
                inverter block locations.
            defect_mapping_dict: Dictionary 
                Dictionary containing mapping between defect ID and defect 
                name.
            metadata: Dictionary
                Dictionary containing information on the system in question, 
                including lat-long coordinates and name.
        Return:
            inv_df: Pandas dataframe
                Dataframe that maps the aerial defect location to the site 
                location.
        """
        # Loop through each inverter block, and get the associated issues from
        # the aerial defect dataframe
        inv_block_defect_list = list()
        used_idx = list()
        for inv_block in site_dict["features"]:
            inv_coords = inv_block["geometry"]["coordinates"]
            inv_coords = list((x[0], x[1]) for x in inv_coords[0])
            inv_block_polygon = Polygon(inv_coords)
            for defect in aerial_defect_dict["features"]:
                defect_coords = defect["geometry"]["coordinates"]
                defect_coords = list((x[0], x[1]) for x in defect_coords[0][0])
                if defect_coords in used_idx:
                    continue
                else:
                    defect_polygon = Polygon(defect_coords)
                    # Check if the defect polygon is in the inverter block
                    # polygon. If so, log it
                    if inv_block_polygon.contains(defect_polygon):
                        try:
                            inv_block_defect_list.append(
                                {"system_id": metadata["system_id"],
                                 "inv_block":
                                 inv_block["properties"]["INVERTER"],
                                 "defect_id":
                                     defect["properties"]["defect_type_id"],
                                 "defect_name": defect_mapping_dict[defect[
                                     "properties"]["defect_type_id"]],
                                 "inv_block_polygon": inv_block_polygon,
                                 "defect_polygon": defect_polygon,
                                 "defect_area": defect_polygon.area})
                        # If there"s no INVERTER property in the geojson, then
                        # default to analyzing for the entire geojson area
                        except:
                            inv_block_defect_list.append(
                                {"system_id": metadata["system_id"], 
                                 "defect_id":
                                 defect["properties"]["defect_type_id"],
                                 "defect_name": defect_mapping_dict[defect[
                                     "properties"]["defect_type_id"]],
                                 "inv_block_polygon": inv_block_polygon,
                                 "defect_polygon": defect_polygon,
                                 "defect_area": defect_polygon.area})
                        used_idx.append(defect_coords)
        inv_df = pd.DataFrame(inv_block_defect_list)  
        return inv_df
    
    def aggregate_defects(self, defect_df, total_module_count):
        """
        Aggregate defects and get the percentage of site/inverter block
        modules that contain a particular defect. This percentage value
        allows for much easier comparison across sites and different
        inverter blocks for a site.
        Parameter:
            defect_df: Pandas dataframe
                Dataframe that contains the mapped aerial defect location
                to the site location.
        Returns
            defect__df: Pandas dataframe
                Datafreme with defect percentage and defect counts for each
                defect in a site or individual inverter block
        """
        if self.by_site:
            defect_df['defect_count'] = defect_df.groupby(
                ["system_id", "defect_id"])['defect_id'].transform("count")
        if self.by_block:
            defect_df['defect_count'] = defect_df.groupby(
                ["inv_block", "defect_id"])['defect_id'].transform("count")
        defect_df['defect_percentage'] = 100 * defect_df[
            'defect_count']/total_module_count
        return defect_df

    def generate_folium_graphic(self, site_name, site_coord, site_dict,
                              aerial_defect_dict,  zoom=15):
        """
        Generates an html of site and aerial defect geojsons using the folium
        package.
        Parameters:
            site_name: str
                Name of site
            site_coord: Tuple of floats
                Site coordinates in (latitude, longitude) format
            site_dict: Dictionary
                Dictionary containing site geojson
            aerial_defect_dict: Dictionary
                Dictionary containing aerial defect analysis geojson
            zoom: Int. default of 15.
                Zoom of map. Please note, large zoom (above 19) will result 
                in the satellite image not displaying
        Returns:
            Folium map object of mapped site.
        """
        # Locate site location
        site_map = folium.Map(location=site_coord,
                              zoom_start=zoom)
        # Add satellite layer
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
            + "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=True).add_to(site_map)

        # Add in site geojson layer
        # If geojson has "INVERTER" property for tooltip
        if self.by_block:
            folium.GeoJson(site_dict,
                           name="Site",
                           style_function=lambda feature:
                               {"fill_color": "green",
                                "color": "green",
                                "weight": 1,
                                "fillOpacity": 0.5},
                           tooltip=folium.GeoJsonTooltip(
                               fields=["INVERTER"],
                               alisaes=["Inverter :"], labels=True),
                           highlight=True).add_to(site_map)

        if self.by_site:
            folium.GeoJson(site_dict,
                           name="Site",
                           style_function=lambda feature:
                               {"fill_color": "green",
                                "color": "green",
                                "weight": 1,
                                "fillOpacity": 0.5},
                           highlight=True).add_to(site_map)

        # Adds aerial defect analysis geojson with "defect_name" tooltip
        folium.GeoJson(aerial_defect_dict,
                       name="Aerial IR Defect",
                       style_function=lambda feature: {"fill_color": "red",
                                                       "color": "red",
                                                       "weight": 1,
                                                       "fillOpacity": 0.5},
                       tooltip=folium.GeoJsonTooltip(fields=["defect_name"],
                                                     alisaes=["Defect Name:"],
                                                     labels=True),
                       highlight=True).add_to(site_map)

        # Save map to HTML
        folium.map.LayerControl("topright").add_to(site_map)
        return site_map

    def visualize_short_term_performance(self, time_series_df,
                                         scan_date,
                                         day_window=7):
        """
        Visualize a time series around a particular aerial scan date. Generates
        a Plotly graphic.
        Parameters:
            time_series_df (dataframe):
                TZ-aware time series dataframe that contains the AC power data 
                we want to plot. Index is datetime values.
            scan_date: Str.
                Aerial scan date.
            day_window: Int. Default 7.
                Specifies the numbers of days before and after the scan date 
                to filter the time series data
        Returns:
            Plotly plot visualizing performance around aerial scan date.
        """
        scan_date = pd.to_datetime(scan_date).tz_localize(
            time_series_df.index.tz)
        scan_end = scan_date + pd.to_timedelta(day_window, unit="d")
        scan_start = scan_date - pd.to_timedelta(day_window, unit="d")
        # Filter the time series to around the site scan period
        time_series_df = time_series_df[
                (time_series_df.index >= scan_start)
                & (time_series_df.index <= scan_end)]
        # Make plotly plots
        fig = px.line(time_series_df,
                      y=time_series_df.columns,
                      title="AC Power Production near Aerial Scan Date"
                      ).update_layout(xaxis_title="Date",
                                             yaxis_title="AC Power")
        # if tz_localize:
        fig.add_vrect(x0=scan_date,
                      x1=scan_date + pd.to_timedelta(1, unit="d"),
                      fillcolor="red",
                      annotation_text="Aerial Scan Date",
                      opacity=0.2)

        fig.update_layout(
            font=dict(
                size=20
            )
        )
        return fig