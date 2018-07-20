import copy
import matplotlib
import matplotlib.cbook
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np

def get_phase_names(phase_name, include_depth_phases = True):
    """
        Called by parse_phase_list to replace e.g. ttall with the relevant phases.
        """
    lphase = phase_name.lower()
    names = []
    if lphase in ("mantle", "outercore", "innercore", "ttall"):
        if lphase in ("mantle",  "ttall"):
            names.extend([ "P", "Pn", "S", "Sn",
                          "PKP",  "SKS", "PKKP", "SKKS", "PcP","ScS",
                          "PP","SS","PPP", "SSS", "SP","PS",
                          "Pdiff","Sdiff"])

        
        if lphase in ("outercore", "ttall"):
            names.extend(["PKP",  "SKS", "PKKP", "SKKS", "PcP","ScS",
                          "PKKKP","SKKKS","PKKKKP", "SKKKKS", "PKKKKKP", "SKKKKKS",
                          "ScP", "SKP", "SKKP", "PKS", "PKKS"])
        
        if lphase in ("innercore", "ttall"):
            names.extend([ "PKiKP", "SKiKS" "PKIKP", "SKIKS", "SKIKP",
                          "PKIKKIKP","SKIKKIKP", "PKIKPPKIKP", "PKIKS",
                          "PKIKKIKS", "SKKS", "SKIKKIKS", "SKSSKS",
                          "SKIKSSKIKS", "PKJKP","SKJKIP"])


        if include_depth_phases:
            phases = copy.copy(names)
            for ph in phases:
                names.extend(["p"+ph,"s"+ph])
            names.extend(["p","s"])

    else:
            names.append(phase_name)
    
    return names



# Pretty paired colors. Reorder to have saturated colors first and remove
# some colors at the end.
cmap = plt.get_cmap('Paired', lut=12)
COLORS = ['#%02d%02d%02d' % tuple(col * 255 for col in cmap(i)[:3])
          for i in range(12)]
COLORS = COLORS[1:][::2][:-1] + COLORS[::2][:-1]


class _SmartPolarText(matplotlib.text.Text):
    """
        Automatically align text on polar plots to be away from axes.
        
        This class automatically sets the horizontal and vertical alignments
        based on which sides of the spherical axes the text is located.
        """
    def draw(self, renderer, *args, **kwargs):
        fig = self.get_figure()
        midx = fig.get_figwidth() * fig.dpi / 2
        midy = fig.get_figheight() * fig.dpi / 2
        
        extent = self.get_window_extent(renderer, dpi=fig.dpi)
        points = extent.get_points()
        
        is_left = points[0, 0] < midx
        is_top = points[0, 1] > midy
        updated = False
        
        ha = 'right' if is_left else 'left'
        if self.get_horizontalalignment() != ha:
            self.set_horizontalalignment(ha)
            updated = True
        va = 'bottom' if is_top else 'top'
        if self.get_verticalalignment() != va:
            self.set_verticalalignment(va)
            updated = True
        
        if updated:
            self.update_bbox_position_size(renderer)
        
        matplotlib.text.Text.draw(self, renderer, *args, **kwargs)



def plot(arrivals, plot_type="spherical", plot_all=True, legend=True,
             label_arrivals=False, ax=None, show=True, plot_one=None, color = None):
        """
        Plot the ray paths if any have been calculated.

        :param plot_type: Either ``"spherical"`` or ``"cartesian"``.
            A spherical plot is always global whereas a Cartesian one can
            also be local.
        :type plot_type: str
        :param plot_all: By default all rays, even those travelling in the
            other direction and thus arriving at a distance of *360 - x*
            degrees are shown. Set this to ``False`` to only show rays
            arriving at exactly *x* degrees.
        :type plot_all: bool
        :param legend: If boolean, specify whether or not to show the legend
            (at the default location.) If a str, specify the location of the
            legend. If you are plotting a single phase, you may consider using
            the ``label_arrivals`` argument.
        :type legend: bool or str
        :param label_arrivals: Label the arrivals with their respective phase
            names. This setting is only useful if you are plotting a single
            phase as otherwise the names could be large and possibly overlap
            or clip. Consider using the ``legend`` parameter instead if you
            are plotting multiple phases.
        :type label_arrivals: bool
        :param ax: Axes to plot to. If not given, a new figure with an axes
            will be created. Must be a polar axes for the spherical plot and
            a regular one for the Cartesian plot.
        :type ax: :class:`matplotlib.axes.Axes`
        :param show: Show the plot.
        :type show: bool

        :returns: The (possibly created) axes instance.
        :rtype: :class:`matplotlib.axes.Axes`
        """
        arrivalstmp = []
        for _i in arrivals:
            if _i.path is None:
                continue
            dist = _i.purist_distance % 360.0
            distance = _i.distance
            if abs(dist - distance) / dist > 1E-5:
                if plot_all is False:
                    continue
                # Mirror on axis.
                _i = copy.deepcopy(_i)
                _i.path["dist"] *= -1.0
            arrivalstmp.append(_i)
        if not arrivalstmp:
            raise ValueError("Can only plot arrivals with calculated ray "
                             "paths.")
        discons = arrivals.model.s_mod.v_mod.get_discontinuity_depths()
        if plot_type == "spherical":
            if not ax:
                plt.figure(figsize=(14, 12))
                if MATPLOTLIB_VERSION < [1, 1]:
                    from .matplotlib_compat import NorthPolarAxes
                    from matplotlib.projections import register_projection
                    register_projection(NorthPolarAxes)
                    ax = plt.subplot(111, projection='northpolar')
                else:
                    ax = plt.subplot(111, polar=True)
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
            ax.set_xticks([])
            ax.set_yticks([])
            intp = matplotlib.cbook.simple_linear_interpolation
            radius = arrivals.model.radius_of_planet
            for _i, ray in enumerate(arrivalstmp):
                # Requires interpolation otherwise diffracted phases look
                # funny.
                if color is None:
                    ax.plot(intp(ray.path["dist"], 100),
                            radius - intp(ray.path["depth"], 100),
                            color=COLORS[_i % len(COLORS)], label=ray.name,
                            lw=1.0)
                else:
                    ax.plot(intp(ray.path["dist"], 100),
                            radius - intp(ray.path["depth"], 100),
                            color=color, label=ray.name,
                            lw=1.0)
            ax.set_yticks(radius - discons)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            # Pretty earthquake marker.
            ax.plot([0], [radius - arrivalstmp[0].source_depth],
                    marker="*", color="#FEF215", markersize=20, zorder=10,
                    markeredgewidth=1.5, markeredgecolor="0.3", clip_on=False)

#            # Pretty station marker.
#            arrowprops = dict(arrowstyle='-|>,head_length=0.8,head_width=0.5',
#                              color='#C95241',
#                              lw=1.5)
#            ax.annotate('',
#                        xy=(np.deg2rad(distance), radius),
#                        xycoords='data',
#                        xytext=(np.deg2rad(distance), radius * 1.02),
#                        textcoords='data',
#                        arrowprops=arrowprops,
#                        clip_on=False)
#            arrowprops = dict(arrowstyle='-|>,head_length=1.0,head_width=0.6',
#                              color='0.3',
#                              lw=1.5,
#                              fill=False)
#            ax.annotate('',
#                        xy=(np.deg2rad(distance), radius),
#                        xycoords='data',
#                        xytext=(np.deg2rad(distance), radius * 1.01),
#                        textcoords='data',
#                        arrowprops=arrowprops,
#                        clip_on=False)
            if label_arrivals:
                name = ','.join(sorted(set(ray.name for ray in arrivalstmp)))
                # We cannot just set the text of the annotations above because
                # it changes the arrow path.
                t = _SmartPolarText(np.deg2rad(distance), radius * 1.07,
                                    name, clip_on=False)
                ax.add_artist(t)


            ax.set_rmax(radius)
            ax.set_rmin(0.0)
            if legend:
                if isinstance(legend, bool):
                    if 0 <= distance <= 180.0:
                        loc = "upper left"
                    else:
                        loc = "upper right"
                else:
                    loc = legend
                plt.legend(loc=loc, prop=dict(size="small"))

        elif plot_type == "cartesian":
            if not ax:
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
            ax.invert_yaxis()
            for _i, ray in enumerate(arrivalstmp):
                ax.plot(np.rad2deg(ray.path["dist"]), ray.path["depth"],
                        color=COLORS[_i % len(COLORS)], label=ray.name,
                        lw=1.0)
            ax.set_ylabel("Depth [km]")
            if legend:
                if isinstance(legend, bool):
                    loc = "lower left"
                else:
                    loc = legend
                ax.legend(loc=loc, prop=dict(size="small"))
            ax.set_xlabel("Distance [deg]")
            # Pretty station marker.
            ms = 14
            station_marker_transform = matplotlib.transforms.offset_copy(
                ax.transData,
                fig=ax.get_figure(),
                y=ms / 2.0,
                units="points")
            ax.plot([distance], [0.0],
                    marker="v", color="#C95241",
                    markersize=ms, zorder=10, markeredgewidth=1.5,
                    markeredgecolor="0.3", clip_on=False,
                    transform=station_marker_transform)
            if label_arrivals:
                name = ','.join(sorted(set(ray.name for ray in arrivals)))
                ax.annotate(name, xy=(distance, 0.0),
                            xytext=(0, ms * 1.5), textcoords='offset points',
                            ha='center', annotation_clip=False)

            # Pretty earthquake marker.
            ax.plot([0], [arrivals[0].source_depth],
                    marker="*", color="#FEF215", markersize=20, zorder=10,
                    markeredgewidth=1.5, markeredgecolor="0.3", clip_on=False)
            x = ax.get_xlim()
            x_range = x[1] - x[0]
            ax.set_xlim(x[0] - x_range * 0.1, x[1] + x_range * 0.1)
            x = ax.get_xlim()
            y = ax.get_ylim()
            for depth in discons:
                if not (y[1] <= depth <= y[0]):
                    continue
                ax.hlines(depth, x[0], x[1], color="0.5", zorder=-1)
            # Plot some more station markers if necessary.
            possible_distances = [_i * (distance + 360.0)
                                  for _i in range(1, 10)]
            possible_distances += [-_i * (360.0 - distance) for _i in
                                   range(1, 10)]
            possible_distances = [_i for _i in possible_distances
                                  if x[0] <= _i <= x[1]]
            if possible_distances:
                ax.plot(possible_distances,  [0.0] * len(possible_distances),
                        marker="v", color="#C95241",
                        markersize=ms, zorder=10, markeredgewidth=1.5,
                        markeredgecolor="0.3", clip_on=False, lw=0,
                        transform=station_marker_transform)

        else:
            raise NotImplementedError
        if show:
            plt.show()
        return ax
