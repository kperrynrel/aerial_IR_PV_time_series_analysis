"""
Master class for fusing aerial IR analysis with time series data.
"""
from shapely import wkt
from shapely.geometry.polygon import Polygon
import pandas as pd
from pyproj import Geod
import ast
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import folium
import matplotlib.pyplot as plt


class DataFusion:

    def __init__(self, defect_category_dict, by_site=False, by_block=False):
        self.defect_category_dict = defect_category_dict
        self.by_site = by_site
        self.by_block = by_block

    def merge_aerial_site_dictionary(self, aerial_defect_dict, site_dict):
        """
        Merge a dictionary containing aerial defect information with a site
        location dictionary. Returns a dataframe where each row is a
        defect found, mapping to a particular area in the site dictionary.
        Parameter:
            aerial_defect_dict (dictionary): dictionary from a geojson file
                    that contains the defect location
            site_dict (dictionary): dictionary from a geojson file that
                    contains the site and all inverter block locations.
        Return:
            inv_df (dataframe): dataframe that maps the aerial defect location
                    to the site location.
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
                                {"inv_block":
                                 inv_block["properties"]["INVERTER"],
                                 "defect_id":
                                     defect["properties"]["defect_type_id"],
                                 "inv_block_polygon": inv_block_polygon,
                                 "defect_polygon": defect_polygon,
                                 "defect_area": defect_polygon.area})
                        # If there"s no INVERTER property in the geojson, then
                        # default to analyzing for the entire geojson area
                        except:
                            inv_block_defect_list.append(
                                {"defect_id":
                                 defect["properties"]["defect_type_id"],
                                 "inv_block_polygon": inv_block_polygon,
                                 "defect_polygon": defect_polygon,
                                 "defect_area": defect_polygon.area})
                        used_idx.append(defect_coords)
        inv_df = pd.DataFrame(inv_block_defect_list)
        return inv_df

    def get_percent_defect_counts(self, defect_df):
        """
        Get a percentage of the site or part of a site that contains a
        particular defect. This percentage value allows for much easier
        comparison across sites and different inverter blocks for a site.
        Parameter:
            defect_df (dataframe): dataframe that contains the mapped aerial
                    defect location to the site location.
        Returns
            defect_area_df (dataframe): datafreme with defect percentage,
                    defect counts, and defect name for each defect in a site
                    or individual inverter block
        """
        defect_category_df = pd.DataFrame(self.defect_category_dict)

        area_list = list()
        inv_area_list = list()
        defect_area_list = list()
        defect_id_list = list()
        for index, row in defect_df.iterrows():
            inv_block_polygon = wkt.loads(str(row["inv_block_polygon"]))
            defect_polygon = wkt.loads(str(row["defect_polygon"]))
            # Get area
            geod = Geod(ellps="WGS84")
            inv_poly_area, _ = geod.geometry_area_perimeter(
                inv_block_polygon)
            defect_poly_area, _ = geod.geometry_area_perimeter(
                defect_polygon)
            if self.by_block:
                defect_pct = (defect_poly_area/inv_poly_area) * 100
                try:
                    area_list.append(
                        {"defect_id": row["defect_id"],
                         "inv_block": row["inv_block"],
                         "inv_area": abs(inv_poly_area),
                         "defect_area": abs(defect_poly_area),
                         "defect_percentage": abs(defect_pct)
                         })
                except:
                    area_list.append(
                        {"defect_id": row["defect_id"],
                         "inv_area": abs(inv_poly_area),
                         "defect_area": abs(defect_poly_area),
                         "defect_percentage": abs(defect_pct)
                         })
            if self.by_site:
                defect_id_list.append(row["defect_id"])
                if inv_poly_area not in inv_area_list:
                    inv_area_list.append(abs(inv_poly_area))
                defect_area_list.append(abs(defect_poly_area))
                site_area = sum(inv_area_list)
        if self.by_site:
            for defect_area, defect_id in zip(defect_area_list,
                                              defect_id_list):
                defect_pct = (defect_area/site_area) * 100
                area_list.append({
                    "defect_id": defect_id,
                    "site_area": site_area,
                    "defect_area": defect_area,
                    "defect_percentage": defect_pct})

        # Make into dataframe
        area_df = pd.DataFrame(area_list)
        area_df["defect_id"] = area_df["defect_id"].astype(float)
        defect_category_df["defect_id"] = defect_category_df[
            "defect_id"].astype(float)
        # Merge dataframes to map defect_id to defect_names in
        defect_area_df = pd.merge(defect_category_df, area_df, on="defect_id")
        return defect_area_df

    def aggregate_defect_areas(self, block_area_df):
        """
        Aggregate up all of the defects of the same class for a particular
        zone/block/site.
        Parameter:
            block_area_df (dataframe): datafreme containing defect percentage,
                    defect counts, and defect name for each defect in a site
                    or individual inverter block.
        Returns
            agg_defect_area_df (dataframe): dataframe containing the aggregated
                    defects of the same class
        """
        if self.by_block:
            try:
                groupby_cols = ["defect_id",
                                "defect_name", "inv_block", "inv_area"]
                agg_defect_area_df = block_area_df.groupby(
                    groupby_cols)[["defect_area", "defect_percentage"]].sum()
                agg_defect_area_df["defect_counts"] = block_area_df.groupby(
                    groupby_cols).size()
            except:
                groupby_cols = ["defect_id", "defect_name", "inv_area"]
                agg_defect_area_df = block_area_df.groupby(
                    groupby_cols)[["defect_area", "defect_percentage"]].sum()
                agg_defect_area_df["defect_counts"] = block_area_df.groupby(
                    groupby_cols).size()
        if self.by_site:
            groupby_cols = ["defect_id", "defect_name", "site_area"]
            agg_defect_area_df = block_area_df.groupby(
                groupby_cols)[["defect_area", "defect_percentage"]].sum()
            agg_defect_area_df["defect_counts"] = block_area_df.groupby(
                groupby_cols).size()
        return agg_defect_area_df.reset_index()

    def aggregate_columns(self, time_series_df, column_list, agg_type="sum"):
        """
        Aggregate across a series of columns in the time series. This is esp
        useful where we need to aggregate power across several inverters
        for a site.
        Parameters:
            time_series_df (dataframe): time series dataframe that contains the
                    the time series data for the site
            column_list (str list): lists of ac power columns to aggregate
            agg_type (str): defaults to sum
        Returns:
            agg_time_series_df (dataframe): time series with aggregated data
            from column list
        """
        agg_values = time_series_df[column_list].agg(agg_type, axis=1)
        agg_time_series_df = pd.DataFrame(agg_values, columns=["agg_ac_power"])
        agg_time_series_df.index = time_series_df.index
        return agg_time_series_df

    def visualize_short_term_performance(self, aggregated_defect_area_df,
                                         time_series_df, time_zone,
                                         short_term_defects_list, site_name,
                                         plot_path, scan_date,
                                         datastream_column_list=None,
                                         day_window=7):
        """
        Visualize a time series around a particular aerial scan date. Generates
        a Plotly graphic.
        Parameters:
            aggregated_defect_area_df (dataframe): aggregated defect dataframe
                    that contains "defect_counts" and "defect_name" columns.
                    If there are inverter block names, then there should
                    be an associated mapped "data_streams" columns
                    in the dataframe.
            time_series_df (dataframe): time series dataframe that contains the
                    the time series data (like AC power) for the site
            time_zone (str): time zone of aerial scans
            short_term_defects_list (str list): lists of short term defects to
                    filter and plot
            site_name (str): site name or location of aerial scans
            plot_path (str): path where the short term performace plots are
                    saved
            scan_date (date or datetime): aerial scan date
            datastream_column_list (str list): defaults to None,
                    list of ac power datastream columns to plot
            day_window (int): defaults to 7,
                    specifies the numbers of days before and after the scan
                    date to filter the time series data
        Returns:
            None
        """
        try:
            # If scan_date is already in datetime format
            scan_end = scan_date + pd.to_timedelta(day_window, unit="d")
            scan_start = scan_date - pd.to_timedelta(day_window, unit="d")
        except:
            scan_date = pd.to_datetime(scan_date)
            scan_end = scan_date + \
                pd.to_timedelta(day_window, unit="d")
            scan_start = scan_date - pd.to_timedelta(day_window, unit="d")
        # Filter the time series to around the site scan period
        try:
            time_series_df = time_series_df[
                (time_series_df.index >= scan_start.tz_localize(time_zone))
                & (time_series_df.index <= scan_end.tz_localize(time_zone))]
            tz_localize = True
        except:
            time_series_df = time_series_df[
                (time_series_df.index >= scan_start)
                & (time_series_df.index <= scan_end)]
            tz_localize = False

        # Filter the defect results to just include stuck trackers and offline
        # strings
        short_term_defects = aggregated_defect_area_df[
            aggregated_defect_area_df["defect_name"].isin(
                short_term_defects_list)]
        short_term_defects = short_term_defects.sort_values(
            by=["defect_counts"], ascending=False)
        if self.by_site:
            # Filter for columns containing "ac" data
            ds_time_series_df = time_series_df[datastream_column_list]
            for index, row in short_term_defects.iterrows():
                # Min-max normalize the data set for anonymity
                scaler = MinMaxScaler()
                ds_time_series_df = pd.DataFrame(
                    scaler.fit_transform(ds_time_series_df),
                    columns=ds_time_series_df.columns,
                    index=ds_time_series_df.index)
                # Write the associated issue type for the location to the
                # plotly graphic
                defect_type, defect_count = (row["defect_name"],
                                             row["defect_counts"])
                # Put all ac power data streams together
                all_ds_df = pd.DataFrame()
                for stream in ds_time_series_df.columns:
                    all_ds_df = pd.concat(
                        [all_ds_df, ds_time_series_df[stream]], axis=0)

                # Make plotly plots
                fig = px.line(ds_time_series_df,
                              y=ds_time_series_df.columns,
                              title=(site_name +
                                     " Short Term Performance: " +
                                     str(defect_count) + " " +
                                     str(defect_type))).update_layout(
                    xaxis_title="Date",
                    yaxis_title="Normalized AC Power")

                # if tz_localize:
                fig.add_vrect(x0=scan_date.tz_localize(time_zone),
                              x1=scan_date.tz_localize(
                                  time_zone) + pd.to_timedelta(1, unit="d"),
                              fillcolor="red",
                              annotation_text="Aerial Scan Date",
                              opacity=0.2)

                fig.update_layout(
                    font=dict(
                        size=20
                    )
                )
                # Display time series only on scan date
                fig.update_xaxes(range=[
                    pd.to_datetime(scan_date).tz_localize(time_zone),
                    pd.to_datetime(scan_date).tz_localize(time_zone) +
                    pd.to_timedelta(1, unit="d")])
                # Make html file
                fig.write_html(plot_path + str(site_name) +
                               "_" + str(defect_type) +
                               "_short_term_performance_plots.html",
                               full_html=False, include_plotlyjs="cdn")

        # loop through each data stream and plot it against the short
        # term defects to get an idea of effects to the signal
        if self.by_block:
            idx_val = 0
            for index, row in short_term_defects.iterrows():
                data_streams = ast.literal_eval(row["data_streams"])
                data_streams = [x.lower() for x in data_streams]
                # Filter the time series to only contain the associated
                # data stream columns
                try:
                    ds_time_series_df = time_series_df[data_streams]
                    # Min-max normalize the data set for anonymity
                    scaler = MinMaxScaler()
                    ds_time_series_df = pd.DataFrame(
                        scaler.fit_transform(ds_time_series_df),
                        columns=ds_time_series_df.columns,
                        index=ds_time_series_df.index)
                    # Write the associated issue type for the location to the
                    # plotly graphic
                    defect_type, defect_count = (row["defect_name"],
                                                 row["defect_counts"])
                    # Make plotly plots
                    fig = px.line(ds_time_series_df,
                                  y=ds_time_series_df.columns,
                                  title=(site_name +
                                         " Short Term Performance: " +
                                         str(defect_count) + " " +
                                         str(defect_type))).update_layout(
                        xaxis_title="Date",
                        yaxis_title="Normalized AC Power")
                    # rename the data streams so they"re anonymized
                    newnames = {}
                    col_idx = 1
                    for stream in data_streams:
                        newnames[stream] = "ac_power_inverter_" + str(col_idx)
                        col_idx += 1
                    fig.for_each_trace(
                        lambda t:
                            t.update(name=newnames[t.name],
                                     legendgroup=newnames[t.name],
                                     hovertemplate=t.hovertemplate.replace(
                                     t.name, newnames[t.name])))
                    # if tz_localize:
                    fig.add_vrect(x0=scan_start.tz_localize(time_zone),
                                  x1=scan_end.tz_localize(
                                      time_zone) +
                                  pd.to_timedelta(1, unit="d"),
                                  fillcolor="red",
                                  annotation_text="Aerial Scan Date",
                                  opacity=0.2)

                    fig.update_layout(
                        font=dict(
                            size=20
                        )
                    )
                    # Display time series only on scan date
                    fig.update_xaxes(range=[
                        pd.to_datetime(scan_date).tz_localize(time_zone),
                        pd.to_datetime(scan_date).tz_localize(time_zone) +
                        pd.to_timedelta(1, unit="d")])
                    # Make html file
                    fig.write_html(plot_path + str(site_name) +
                                   "_" + str(idx_val) + "_" + str(defect_type)
                                   + "_short_term_performance_plots.html",
                                   full_html=False, include_plotlyjs="cdn")
                    idx_val += 1
                except Exception as e:
                    print(e)
                    print(
                        "Couldn't run for the following data streams: " +
                        str(data_streams))

    def generate_geojson_html(self, site_name, site_coord, site_dict,
                              aerial_defect_dict, result_file_path, zoom=15):
        """
        Generates an html of site and aerial defect geojsons using the folium
        package.
        Parameters:
            site_name (str): name of site
            site_coord (float tuple): site coordinates in
                    (latitude, longitude) format
            site_dict (str): dictionary cointaining site geojson
            aerial_defect_dict (str): dictionary containing aerial defect
                    analysis geojson
            by_site (boolean): defaults to False,
                    if True, does not label the inverter block names
            by_block (boolean): defaults to False,
                    if True, labels the inverter block names
            result_file_path (str): location of saved html file
            zoom (int): defaults to 15,
                    zoom of map,
                    note: large zoom (above 19) will result in the satellite
                    image not displaying
        Returns:
            None
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
        save_path = str(result_file_path) + str(site_name) + "_folium_map.html"
        site_map.save(save_path)

    def generate_histogram(self, agg_defect_df, site_name, plot_path):
        """
        Generates a short term and long term histogram of defect counts
        and defect type.
        Parameters:
            agg_defect_df (dataframe): aggregated defect dataframe that
                    contains the aggregated defects of the same class. It has
                    columns "defect_name" and "defect_counts".
            site_name (str): name of site
            plot_path (str): path where the histogram plots are saved
        Returns:
            None
        """
        if self.by_site:
            column = "site_area"
        if self.by_block:
            column = "inv_block"
        for block in agg_defect_df[column].unique():
            filtered_df = agg_defect_df[agg_defect_df[column] == block]
            filtered_df.loc[filtered_df["defect_name"] ==
                            "Isolated/ Underperforming Module",
                            "defect_name"] = "Underperforming Module"

            color_dict = {"String Off-line": "red",
                          "Missing Module": "blue",
                          "Misaligned Modules": "lightblue",
                          "Suspected PID": "purple",
                          "Soiling": "yellow",
                          "Single Hotspot <10C": "green",
                          "Multi-Hotspots <10C": "orange",
                          "Single Hotspot >20C": "turquoise",
                          "Underperforming Module": "pink",
                          "Single Hotspot 10C-20C": "lightgreen",
                          "Broken Glass": "lightsalmon",
                          "Hotspots": "gray",
                          "Diode Bypass": "orchid",
                          "Sub-string short circuit": "dodgerblue"}

            sys_df = filtered_df.drop_duplicates()
            sys_df = sys_df.sort_values(by="defect_counts",
                                        ascending=False)
            short_term_issues = ["Misaligned Modules",
                                 "String Off-line",
                                 "Missing Module",
                                 "Soiling"]
            short_term_counts = list(sys_df[sys_df["defect_name"].isin(
                short_term_issues)]["defect_counts"])
            short_term_issue_present = list(sys_df[sys_df["defect_name"].isin(
                short_term_issues)]["defect_name"])

            long_term_issues = ["Diode Bypass",
                                "Underperforming Module",
                                "Single Hotspot <10C",
                                "Sub-string short circuit",
                                "Broken Glass",
                                "Multi-Hotspots <10C",
                                "Single Hotspot 10C-20C",
                                "Suspected PID",
                                "Single Hotspot >20C"]
            long_term_counts = list(sys_df[sys_df["defect_name"].isin(
                long_term_issues)]["defect_counts"])
            long_term_issue_present = list(sys_df[sys_df["defect_name"].isin(
                long_term_issues)]["defect_name"])

            # Subplots for defect counts
            fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[4.5, 10])

            # Plot short term performance
            short_term_colors = [color_dict[x]
                                 for x in short_term_issue_present]
            ax1.bar(short_term_issue_present, short_term_counts,
                    label=short_term_issue_present, color=short_term_colors)
            ax1.set_ylabel("Number Module Defects")
            ax1.set_title(
                "Total Module Defect Counts, \n Short-Term/Other Defects",
                y=1.02)
            ax1.legend(title="Defect", loc="upper left",
                       fancybox=True, shadow=True)
            ax1.get_legend().remove()
            # Add the counts on top the bars
            for p in ax1.patches:
                ax1.annotate(str(p.get_height()),
                             xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                             xytext=(0, 10),
                             textcoords="offset points",
                             ha="center",
                             va="top")
            ax1.tick_params(axis="x", rotation=70)
            ax1.set_ylim(
                [0, (max(short_term_counts) + max(short_term_counts)/10)])
            plt.tight_layout()
            # Save long term plot
            if self.by_site:
                plt.savefig(plot_path + str(site_name) +
                            "_short_term_histogram.png")
            if self.by_block:
                plt.savefig(plot_path + str(site_name) + "_" +
                            str(block) + "_short_term_histogram.png")

            # Plot long term performance
            long_term_colors = [color_dict[x] for x in long_term_issue_present]
            ax2.bar(long_term_issue_present, long_term_counts,
                    label=long_term_issue_present, color=long_term_colors)
            ax2.set_title(
                "Total Module Defect Counts, \n Long-Term or BoS Defects",
                y=1.02)
            ax2.legend(title="Defect", loc="upper left",
                       fancybox=True, shadow=True)
            ax2.get_legend().remove()

            # Add the counts on top the bars
            for p in ax2.patches:
                ax2.annotate(str(p.get_height()),
                             xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                             xytext=(0, 10),
                             textcoords="offset points",
                             ha="center",
                             va="top")
            ax2.tick_params(axis="x", rotation=70)
            ax2.set_ylim(
                [0, (max(long_term_counts) + max(long_term_counts)/10)])
            plt.tight_layout()
            # Save long term plot
            if self.by_site:
                plt.savefig(plot_path + str(site_name) +
                            "_long_term_histogram.png")
            if self.by_block:
                plt.savefig(plot_path + str(site_name) + "_" +
                            str(block) + "_long_term_histogram.png")
            plt.show()
