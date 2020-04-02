import cairosvg
import csv
import datetime
import json
import tempfile

from openpyxl.styles import Alignment
from openpyxl.styles import Font
from openpyxl import Workbook
from Products.Five.browser import BrowserView
from StringIO import StringIO
from zope.component import queryMultiAdapter


class ExportCSV(BrowserView):
    """ Export to CSV
    """
    def write_headers(self, sheet, headers, row):
        col = 1
        for h in headers:
            sheet.cell(row=row, column=col).value = h
            sheet.cell(row=row, column=col).font = Font(bold=True)
            col += 1

    def write_encoded_val(self, headers, sheet, encoded, row):
        for idx, h in enumerate(headers):
            sheet.cell(row=row, column=idx + 1).value = encoded[h]

    def datapoints(self, sheet, chart_data):
        """ Export single dimension series to CSV
        """
        try:
            if len(chart_data) < 1:
                return ""
        except:
            return ""

        headers = ['series', 'name', 'code', 'y']
        sheet.cell(row=1, column=1).value = 'Data extracted'
        sheet.cell(row=1, column=1).font = Font(bold=True)
        sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=4)

        self.write_headers(sheet, headers, row=2)

        row = 3
        for series in chart_data:
            for point in series['data']:
                encoded = {}
                encoded['series'] = series.get('name', '-')
                encoded['name'] = point.get('name', '-')
                for key in headers[1:]:
                    encoded[key] = unicode(point.get(key, '-')).encode('utf-8')
                    if point.get('isNA', False):
                        encoded['y'] = None

                self.write_encoded_val(headers, sheet, encoded, row)
                row += 1

    def datapoints_n(self, sheet, chart_data):
        """ Export multiple dimension series to CSV
        """
        try:
            if len(chart_data) < 1:
                return ""
        except:
            return ""

        coords = {'x', 'y', 'z'}
        keys = set(chart_data[0][0].get('data', [{}])[0].keys())

        headers = ['series', 'name', 'x', 'y', 'z']

        if keys.intersection(coords) != coords:
            headers = ['series', 'name', 'x', 'y']

        self.write_headers(sheet, headers, row=1)

        row = 2
        for series in chart_data:
            for point in series:
                encoded = {}
                encoded['series'] = point['name']
                for data in point['data']:
                    for key in headers[1:]:
                        encoded[key] = unicode(data[key]).encode('utf-8')

                    for idx, h in enumerate(headers):
                        sheet.cell(row=row, column=idx + 1).value = encoded[h]

                    row += 1

    def datapoints_profile(self, sheet, chart_data):
        headers = ['name', 'eu', 'original']
        extra_headers = ['period']
        self.write_headers(sheet, extra_headers + headers, row=1)

        row = 2
        for series in chart_data:
            for point in series['data']:
                encoded = {}
                for key in headers:
                    encoded[key] = unicode(point[key]).encode('utf-8')
                period = point['attributes']['time-period']['notation']
                encoded['period'] = unicode(period).encode('utf-8')
               
                self.write_encoded_val(
                    extra_headers + headers, sheet, encoded, row
                )
                row += 1

    def datapoints_profile_table(self, sheet, chart_data):
        row = 1
        for series in chart_data:
            encoded = {}
            latest = series['data']['latest']
            years = [
                '%s' % (latest-3),
                '%s' % (latest-2),
                '%s' % (latest-1),
                '%s' % (latest)
            ]

            headers = (['country', 'indicator', 'breakdown', 'unit'] + years +
                       ['EU28 value %s' %latest, 'rank'])

            self.write_headers(sheet, headers, row=row)

            row += 1
            encoded['country'] = series['data']['ref-area']['label']
            for ind in series['data']['table'].values():
                encoded['indicator'] = unicode(ind['indicator']).encode(
                    'utf-8'
                )
                encoded['breakdown'] = unicode(ind['breakdown']).encode(
                    'utf-8'
                )
                encoded['unit'] = unicode(ind['unit-measure']).encode('utf-8')
                for year in years:
                    encoded[year] = unicode(ind.get(year, '-')).encode('utf-8')
                #encoded['%s' %latest] = unicode(ind.get('%s' %latest, '-')).encode('utf-8')
                encoded['EU28 value %s' %latest] = unicode(
                        ind.get('eu', '-')).encode('utf-8')
                rank = ind.get('rank', '-')
                if rank == 0:
                    rank = '-'
                encoded['rank'] = unicode(rank).encode('utf-8')
                
                self.write_encoded_val(headers, sheet, encoded, row)
                row += 1

    def datapoints_profile_polar(self, sheet, chart_data):
        headers = ['country', 'category', 'indicator', 'breakdown',
                   'unit', 'eu', 'original', 'period']

        self.write_headers(sheet, headers, row=1)

        row = 2
        for series in chart_data:
            for point in series['data']:
                encoded = {}
                encoded['country'] = unicode(
                    point['attributes']['ref-area']['notation']
                ).encode('utf-8')
                encoded['category'] = unicode(point['title']).encode('utf-8')
                encoded['indicator'] = unicode(
                    point['attributes']['indicator']['notation']
                ).encode('utf-8')
                encoded['breakdown'] = unicode(
                    point['attributes']['breakdown']['notation']
                ).encode('utf-8')
                encoded['unit'] = unicode(
                    point['attributes']['unit-measure']['notation']
                ).encode('utf-8')
                encoded['eu'] = unicode(
                    point['attributes']['eu']
                ).encode('utf-8')
                encoded['original'] = unicode(
                    point['attributes']['original']
                ).encode('utf-8')
                encoded['period'] = unicode(
                    point['attributes']['time-period']['notation']
                ).encode('utf-8')
                
                self.write_encoded_val(headers, sheet, encoded, row)
                row += 1

    def write_general_sheet(self, sheet, data):
        sheet.cell(row=1, column=1).value = 'Chart title'

        chart_url = data.get('chart-url', None)
        chart_title = data.get('chart-title', '-')
        chart_subtitle = data.get('chart-subtitle', None)

        if chart_subtitle:
            chart_title = '{} ({})'.format(chart_title, chart_subtitle)

        if chart_url:
            sheet.cell(row=1, column=2).hyperlink = chart_url
        sheet.cell(row=1, column=2).value = chart_title

        sheet.cell(row=2, column=1).value = 'Source dataset'
        sheet.cell(row=2, column=2).value = data.get('source-dataset', '-')

        sheet.cell(row=3, column=1).value = 'Extraction-Date'
        sheet.cell(row=3, column=2).value = datetime.date.today()

        sheet.cell(row=4, column=1).value = 'List of available indicators'
        sheet.cell(row=4, column=2).value = data.get('indicators_details_url')

    def write_applied_filters_sheet(self, sheet, data):
        header = [u'Filter', u'Notation', u'Label', u'Definition']
        self.write_headers(sheet, header, row=1)

        row = 2
        for anno in data['annotations']:
            if anno['notation'] not in data['filter-labels'].keys():
                sheet.cell(row=row, column=1).value = anno['filter_label']
                sheet.cell(row=row, column=2).value = anno['notation']
                sheet.cell(row=row, column=3).value = anno['label']
                sheet.cell(row=row, column=4).value = anno.get(
                    'definition', None
                )

            row += 1

        for notation, filter in data['filter-labels'].items():
            sheet.cell(row=row, column=1).value = filter['filter-label']

            row_incr = 0
            if type(filter['label-col']) is dict:
                sheet.merge_cells(
                    start_row=row, start_column=1,
                    end_row=row + len(filter['label-col']) - 1, end_column=1
                )

                for notation, label in filter['label-col'].items():
                    sheet.cell(row=row+row_incr, column=2).value = notation
                    sheet.cell(row=row+row_incr, column=3).value = label
                    row_incr += 1
                row += row_incr
            else:
                sheet.cell(row=row, column=2).value = notation
                sheet.cell(row=row, column=3).value = filter['label-col']

                row += 1

    def write_observation_data_sheet(self, sheet, chart_data):
        formatters = {
            'scatter': self.datapoints_n,
            'bubbles': self.datapoints_n,
            'country_profile_bar': self.datapoints_profile,
            'country_profile_table': self.datapoints_profile_table,
            'country_profile_polar_polar': self.datapoints_profile_polar,
        }

        chart_type = self.request.form.pop('chart_type')

        formatter = formatters.get(chart_type, self.datapoints)
        formatter(sheet, chart_data)

    def make_sheet(self, wb, name, data, sheet_fct):
        last_sheet = name == 'Observation Data'
        filters_sheet = name == 'Applied Filters'

        if not (last_sheet or filters_sheet):
            sheet = wb.active
            sheet.title = name
        else:
            sheet = wb.create_sheet(name)

        sheet_fct(sheet, data)
        self.apply_styles(sheet)

    @staticmethod
    def apply_styles(sheet, col_width=50, alignment='center'):
        # Set cell width, center cells
        for col in sheet.columns:
            idx = col[0].column
            sheet.column_dimensions[idx].width = col_width
            for cell in col:
                cell.alignment = Alignment(
                    horizontal='left',
                    vertical=alignment,
                    wrapText=True
                )
        # Set cell height
        for row in sheet.rows:
            idx = row[0].row
            sheet.row_dimensions[idx].height = 30

    def export(self):
        if not self.request.form.get('chart_data'):
            return

        metadata = {}
        if self.request.form.get('metadata'):
            metadata = json.loads(self.request.form.pop('metadata'))

        annotations = {}
        if self.request.form.get('annotations'):
            annotations = json.loads(self.request.form.pop('annotations'))

        chart_data = json.loads(self.request.form.pop('chart_data'))

        extra_info = self.request.form.pop('chart_filter_labels')
        if type(extra_info) is str:
            extra_info = json.loads(extra_info)
        else:
            extra_info = json.loads(extra_info[0])

        general_info_data = {
            u'chart-title': metadata['chart-title'],
            u'chart-subtitle': extra_info.get('chart_subtitle'),
            u'chart-url': metadata['chart-url'],
            u'source-dataset': metadata['source-dataset'],
            u'indicators_details_url': annotations['indicators_details_url']
        }

        applied_filters_data = {
            u'filter-labels': extra_info['filters'],
            u'annotations': annotations['blocks']
        }

        wb = Workbook()
        self.make_sheet(
            wb, 'General Information', general_info_data,
            self.write_general_sheet
        )
        self.make_sheet(
            wb, 'Applied Filters', applied_filters_data,
            self.write_applied_filters_sheet
        )
        self.make_sheet(
            wb, 'Observation Data', chart_data,
            self.write_observation_data_sheet
        )

        return self.download_xls(wb)

    def convert_to_csv(self, workbook, stream):
        sheets = workbook.worksheets

        writer = csv.writer(stream)
        for sheet in sheets:
            for row in sheet.rows:
                data_row = map(lambda x: x.value, row)
                writer.writerow(data_row)

        return stream

    def download_xls(self, wb):
        to_xlsx = self.request.form.get('format') == 'xlsx'
        title = self.context.getId().replace(" ", "_")
        timestamp = datetime.date.today().strftime("%d_%m_%Y")

        filename = title + '_' + str(timestamp)

        if to_xlsx:
            stream = StringIO()
            wb.save(stream)
            stream.seek(0)

            self.request.response.setHeader(
                'Content-Type', 'application/vnd.ms-excel; charset=utf-8')
            self.request.response.setHeader(
                'Content-Disposition',
                'attachment; filename="%s.xlsx"' % filename)

            return stream.read()
        else:
            output_stream = self.request.response

            self.request.response.setHeader(
                'Content-Type', 'application/csv; charset=utf-8')
            self.request.response.setHeader(
                'Content-Disposition',
                'attachment; filename="%s.csv"' % filename)

            self.convert_to_csv(wb, output_stream)
            return self.request.response


class ExportRDF(BrowserView):
    """ Export to RDF
    """
    def datapoints(self, points):
        """ xxx
        """
        return "Not implemented error"

    def datapoints_xy(self, points):
        """
        """
        return "Not implemented error"

    def export(self):
        """ Export to csv
        """
        options = self.request.form.get('options', "{}")
        method = self.request.form.get('method', 'datapoints')
        formatter = getattr(self, method, None)

        self.request.form = json.loads(options)
        points = queryMultiAdapter((self.context, self.request), name=method)

        self.request.response.setHeader(
            'Content-Type', 'application/rdf+xml')
        self.request.response.setHeader(
            'Content-Disposition',
            'attachment; filename="%s.rdf"' % self.context.getId())

        if not points:
            return ""

        if formatter:
            return formatter(points)

        return ""


def cairosvg_surface_color(string=None, opacity=1):
    """
    Replace ``string`` representing a color by a RGBA tuple.
    Function overwritten to catch exceptions at line 295.
    """
    from cairosvg.surface.colors import COLORS

    if not string or string in ("none", "transparent"):
        return (0, 0, 0, 0)

    string = string.strip().lower()

    if string in COLORS:
        string = COLORS[string]

    if string.startswith("rgba"):
        r, g, b, a = tuple(
            float(i.strip(" %")) * 2.55 if "%" in i else float(i)
            for i in string.strip(" rgba()").split(","))
        return r / 255, g / 255, b / 255, a * opacity
    elif string.startswith("rgb"):
        r, g, b = tuple(
            float(i.strip(" %")) / 100 if "%" in i else float(i) / 255
            for i in string.strip(" rgb()").split(","))
        return r, g, b, opacity

    if len(string) in (4, 5):
        string = "#" + "".join(2 * char for char in string[1:])
    if len(string) == 9:
        try:
            opacity *= int(string[7:9], 16) / 255
        except:
            pass

    try:
        plain_color = tuple(
            int(value, 16) / 255. for value in (
                string[1:3], string[3:5], string[5:7]))
    except ValueError:
        # Unknown color, return black
        return (0, 0, 0, 1)
    else:
        return plain_color + (opacity,)

class ExportSvg(BrowserView):
    def export(self):
        """
        Simply serve the png submitted as request parameter.
        This is needed only for the Content-Disposition header.
        """
        svg = self.request.get('svg')
        filename = self.request.get('filename', 'chart');
        self.request.response.setHeader(
            'Content-Type', 'image/svg+xml')
        self.request.response.setHeader(
            'Content-Disposition',
            'attachment; filename="' + filename + '.svg"')
        self.request.response.write(svg)
        return self.request.response

class SvgToPng(BrowserView):
    def convert(self):
        """
        Converts a svg to png and http returns the png.
        """
        svg = self.request.get('svg')
        filename = self.request.get('filename', 'chart');
        png_file = tempfile.TemporaryFile(mode='w+b')

        cairosvg.surface.color = cairosvg_surface_color
        cairosvg.svg2png(bytestring=svg, write_to=png_file)

        self.request.response.setHeader(
            'Content-Type', 'image/png')
        self.request.response.setHeader(
            'Content-Disposition',
            'attachment; filename="' + filename + '.png"')

        png_file.flush()
        png_file.seek(0)

        self.request.response.write(png_file.read())

        return self.request.response


class UnicodeWriter(object):
    def __init__(self, *args, **kwargs):
        self.writer = csv.writer(*args, **kwargs)

    def writerow(self, row):
        self.writer.writerow([
            x.encode('utf-8') if isinstance(x, unicode) else x for x in row])
