# npmi cycle rank 4: 24 latents, lifetime 0.577, score 13.84, chordless True, beats nulls False

Latents in cycle order. Label each with a short description; note which edge looks like the closing (rarest) edge.

## latent 23609 (rate 5.88e-05)  label: ____
- `'defined(CONFIG_FB) && !defined(CONFIG_HAS_EARLYSUSPEND))\nstatic const struct'` (8.8)
- `'\t*rmi4_data)\n{\n\treturn 0;\n};\n#endif\n\nstatic const struct'` (8.1)
- `'.resume  = synaptics_rmi4_resume,\n};\n#else\nstatic const struct'` (8.1)
- `'\n\t\t\t\tstruct file * filp);\n\nconst struct'` (7.3)
- `'\n#ifndef __HASH_STRING__\n#define __HASH_STRING__\n  template <>\n    struct'` (6.8)
- `'_probe(struct i2c_client *client,\n\t\tconst struct'` (6.3)
- `'input_close,\n\t.trigger = snd_virmidi_input_trigger,\n};\n\nstatic struct'` (6.1)
- `'.h"\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n\nextern const struct'` (6.1)

## latent 4565 (rate 3.93e-05)  label: ____
- `'_suspend(struct device *dev);\n\nstatic int synaptics_rmi4_resume(struct'` (17.4)
- `' turns off the power to the sensor.\n */\nstatic int synaptics_rmi4_suspend(struct'` (17.2)
- `' and starts finger data\n * acquisition.\n */\nstatic int synaptics_rmi4_resume(struct'` (17.1)
- `' to the\n * assigned page.\n */\nstatic int synaptics_rmi4_set_page(struct'` (17.1)
- `'\n * open rawmidi handle for output\n */\nstatic int snd_virmidi_output_open(struct'` (17.1)
- `'\n * close rawmidi handle for input\n */\nstatic int snd_virmidi_input_close(struct'` (16.9)
- `'\n * trigger rawmidi stream for input\n */\nstatic void snd_virmidi_input_trigger(struct'` (16.9)
- `'\n * open rawmidi handle for input\n */\nstatic int snd_virmidi_input_open(struct'` (16.8)

## latent 9324 (rate 6.13e-04)  label: ____
- `'4_i2c_read(struct synaptics_rmi4_data *rmi4_data,'` (9.1)
- `'4_regulator_lpm(struct synaptics_rmi4_data *rmi4_data,'` (8.7)
- `' *rmi4_data,\n\t\tstruct synaptics_rmi4_fn *fhandler,\n\t\t'` (8.6)
- `'f1a_alloc_mem(struct synaptics_rmi4_data *rmi4_data,'` (8.6)
- `'_fn *fhandler,\n\t\tstruct synaptics_rmi4_fn_desc *fd,\n\t\t'` (8.6)
- `' rb tree.\n */\nint ext3_htree_store_dirent(struct file *dir_file,'` (8.6)
- `' *rmi4_data,\n\t\tstruct synaptics_rmi4_fn *fhandler,\n\t\t'` (8.6)
- `'rmi4_set_page(struct synaptics_rmi4_data *rmi4_data,'` (8.4)

## latent 8778 (rate 1.05e-03)  label: ____
- `'\tcontainer_of(work, struct synaptics_rmi4_data,\n\t\t\tdet_work.work'` (5.2)
- `'i4_data->det_work);\n\tflush_workqueue(rmi4_data->det_work'` (5.2)
- `'i4_data->det_work);\n\tflush_workqueue(rmi4_data->det_work'` (5.1)
- `' = snd_virmidi_input_close,\n\t.trigger'` (5.1)
- `' = device;\n\trdev->client = -1;\n\trwlock_init(&rdev->filelist'` (5.0)
- `'i = substream->runtime->private_data;\n\n\tif (up) {\n\t\tvmidi->trigger'` (4.9)
- `'page_cache_sync_readahead(\n\t\t\t\t\tsb->s_bdev->bd_inode->i_mapping'` (4.9)
- `')index << PAGE_CACHE_SHIFT;\n\t\t\tbh = ext3_bread(NULL, inode, blk'` (4.8)

## latent 3889 (rate 1.19e-03)  label: ____
- `'enable_irq:\n\tcancel_delayed_work_sync(&rmi4_data->det_work);'` (8.0)
- `'data;\n\trdev->flags &= ~SNDRV_VIRMIDI_SUBSCRIBE;'` (7.8)
- `'exit:\n\tmutex_unlock(&(rmi4_data->rmi4_io_ctrl_mutex));'` (7.7)
- `'\n\t\treturn;\n\t}\n\n\tdevice_ctrl.sleep_mode = SENSOR_SLEEP;'` (7.7)
- `'))\n\t\treturn rc;\n\n\trmi4_pdata->reset_delay = RESET_DELAY;'` (7.6)
- `':\n\tmutex_unlock(&(rmi4_data->rmi4_io_ctrl_mutex));'` (7.4)
- `'_pull_up)\n\t\tregulator_put(rmi4_data->vcc_i2c);'` (7.3)
- `'EFAULT;\n\trdev->flags |= SNDRV_VIRMIDI_SUBSCRIBE;'` (7.2)

## latent 24893 (rate 5.44e-04)  label: ____
- `'itance_button_map(rmi4_data, fhandler);\n\tif (retval < 0)\n\t\t'` (8.9)
- `' (rdev->seq_mode == SNDRV_VIRMIDI_SEQ_DISPATCH)\n\t\t'` (8.6)
- `'client->dev,\n\t\t\t"Regulator vdd enable failed rc=%d\\n",\n\t\t\tretval);\n\t\t'` (8.5)
- `':\n\tif (gpio_is_valid(rmi4_data->board->reset_gpio))\n\t\t'` (8.4)
- `'_rmi4_parse_dt(&client->dev, platform_data);\n\t\tif (retval)\n\t\t\t'` (8.2)
- `'_page(rmi4_data, addr);\n\tif (retval != PAGE_SELECT_LEN)\n\t\t'` (7.9)
- `' fb_register_client(&rmi4_data->fb_notif);\n\tif (retval)\n\t\t'` (7.9)
- `'data(dev);\n\n\tif (sscanf(buf, "%u", &input) != 1)\n\t\t'` (7.9)

## latent 7719 (rate 5.02e-04)  label: ____
- `'\tmutex_unlock(&rmi->support_fn_list_mutex);\n\n\tif (f1a) {'` (8.7)
- `'_virmidi *vmidi = substream->runtime->private_data;\n\n\tif (up) {'` (8.7)
- `'\tint count, res;\n\tunsigned char buf[32], *pbuf;\n\n\tif (up) {'` (8.6)
- `'{\n\tint retval = 0;\n\tunsigned char *intr_status;\n\n\tif (enable) {'` (8.6)
- `'->vdd);\n\tif (rmi4_data->board->i2c_pull_up) {'` (8.3)
- `'_find_property(np, "synaptics,button-map", NULL);\n\tif (prop) {'` (8.3)
- `' snd_virmidi_dev *rdev)\n{\n\tif (rdev->client >= 0) {'` (8.2)
- `'\tmutex_unlock(&rmi->support_fn_list_mutex);\n\n\tif (data_sources) {'` (8.1)

## latent 32430 (rate 2.78e-04)  label: ____
- `'fhandler = kzalloc(sizeof(**fhandler), GFP_KERNEL);\n\tif (!(*fhandler))\n\t'` (14.2)
- `'fingers_to_process * size_of_2d_data);\n\tif (retval < 0)\n\t'` (13.9)
- `'query_5.data,\n\t\t\tsizeof(query_5.data));\n\tif (retval < 0)\n\t'` (13.7)
- `'\n\t\t\tf01_query,\n\t\t\tsizeof(f01_query));\n\tif (retval < 0)\n\t'` (13.7)
- `'of_query8,\n\t\t\tsizeof(size_of_query8));\n\tif (retval < 0)\n\t'` (13.7)
- `' = fb_register_client(&rmi4_data->fb_notif);\n\tif (retval)\n\t'` (13.6)
- `'addr.ctrl_base,\n\t\t\tcontrol,\n\t\t\tsizeof(control));\n\tif (retval < 0)\n\t'` (13.6)
- `'status_reg,\n\t\t\tnum_of_finger_status_regs);\n\tif (retval < 0)\n\t'` (13.5)

## latent 24810 (rate 1.55e-04)  label: ____
- `'", args)\n\t}\n\n\tclient, _, err := f.Clients()\n\tif err != nil {'` (9.7)
- `':\n\t\tparts := strings.Split(args[0], ":")\n\t\tif len(parts) != 2 {'` (9.0)
- `'ateRequest(fakeReq)\n\ttokenReview.Status.Authenticated = ok\n\tif err != nil {'` (8.9)
- `'UserName\n\n\tactualIdentity, err := o.IdentityClient.Create(identity)\n\tif err != nil {'` (8.5)
- `' info to API calling convention\n\tinfop, err := convertDriverInfo(info)\n\tif err != nil {'` (8.5)
- `'\n\tif err != nil {'` (8.5)
- `'(errChan)\n\t\tresponse, err := client.CreateVpc(request)\n\t\tif err != nil {'` (8.4)
- `' (o *CreateIdentityOptions) Validate() error {\n\tif len(o.ProviderName) == 0 {'` (8.3)

## latent 5066 (rate 2.82e-04)  label: ____
- `'\t}\n\n\treturn nil\n}\n\nfunc (o *CreateIdentityOptions) Validate() error {\n\t'` (10.3)
- `'o.ProviderName) == 0 {\n\t\treturn fmt.Errorf("provider name is required")\n\t}\n\t'` (9.9)
- `'LogLevel\n\tstr   string\n}\n\nfunc (l line) append(s string) line {\n\t'` (9.6)
- `' with\n// the given id.\nfunc UnprepareLayer(info DriverInfo, layerId string) error {\n\t'` (9.4)
- `'output")\n\n\to.Printer = func(obj runtime.Object, out io.Writer) error {\n\t\t'` (9.4)
- `' starting\n}\n\nfunc (s *status) isInMergePeriod(now time.Time) bool {\n\t'` (9.2)
- `'quest.Context, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {\n\t'` (9.2)
- `'.NewBadRequest(fmt.Sprintf("not a TokenReview: %#v", obj))\n\t}\n\t'` (9.2)

## latent 15011 (rate 2.33e-05)  label: ____
- `'_NAME>:<PROVIDER_USER_NAME> is required")\n\tcase 1:\n\t\tparts := strings'` (3.0)
- `'REST) Create(ctx genericapirequest.Context, obj runtime.Object, includeUninitialized bool) (runtime'` (3.0)
- `' defaultMergePeriod.\ntype MergeLogger struct {\n\t*capnslog.PackageLogger\n\n\tmu      sync'` (2.5)
- `'(\n\t\t&cache.ListWatch{\n\t\t\tListFunc: func(options v1.ListOptions) (runtime'` (2.4)
- `'Format = cmdutil.GetFlagString(cmd, "output")\n\n\to.Printer = func(obj runtime'` (2.3)
- `'TokenReview{}\n}\n\nfunc (r *REST) Create(ctx genericapirequest.Context, obj runtime'` (2.3)
- `'GetFlagString(cmd, "output")\n\n\to.Printer = func(obj runtime.Object, out io'` (2.2)
- `' a new identity\nfunc NewCmdCreateIdentity(name, fullName string, f *clientcmd.Factory, out io'` (2.1)

## latent 23520 (rate 4.89e-05)  label: ____
- `' <-chan error) {\n\tresponseChan := make(chan *CreateVpcResponse, 1)\n\terrChan'` (1.5)
- `'\n\tif err != nil {\n\t\tdefer close(result)\n\t\tcallback(nil, err)\n\t\tresult'` (1.3)
- `'\n * synaptics_rmi4_irq()\n *\n * Called by the kernel when an interrupt occurs ('` (0.9)
- `' ads.\n\nHere\'s the full text:\n\n"Like many of the millions of people who have seen your toy'` (0.9)
- `' days and, moreover, to 1 (0--2), 4 (1--8) and 8 (4--13)'` (0.9)
- `' 10‐item version, and modeled as a dichotomous variable (cutoff ≤1.85). Physical activity is considered'` (0.9)
- `' in 2007. Other publications describe the household survey in more detail.[@b17] [@b18] We categorised communities as'` (0.8)
- `', with combined diagnosis and therapy costs only moderately higher than for the standard strategy. It is a matter of debate whether a diagnostic'` (0.8)

## latent 18895 (rate 1.09e-05)  label: ____
- `', STRICT LIABILITY, OR TORT (INCLUDING\n * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE'` (13.7)
- `' WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n// OUT OF OR IN CONNECTION WITH THE SOFTWARE'` (13.7)
- `'\n.\\" LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY\n.\\" OUT OF THE USE OF THIS SOFTWARE'` (13.2)
- `' STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)\n\tARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE'` (13.0)
- `' AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n * OUT OF OR IN CONNECTION WITH THE SOFTWARE'` (12.8)
- `' FROM,\n// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n// THE SOFTWARE'` (12.7)
- `' FROM,\n * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n * THE SOFTWARE'` (12.5)
- `' and this permission notice shall be included in\n * all copies or substantial portions of the Software.\n *\n * THE SOFTWARE'` (12.0)

## latent 7762 (rate 1.55e-04)  label: ____
- `' this permission notice shall be included in\n * all copies or substantial portions of the Software.\n *\n * THE SOFTWARE IS'` (11.0)
- `' to endorse or promote products derived from this software\n.\\"    without specific prior written permission.\n.\\"\n.\\" THIS SOFTWARE IS'` (10.9)
- `' this permission notice shall be included in\n// all copies or substantial portions of the Software.\n//\n// THE SOFTWARE IS'` (10.4)
- `' used to endorse or promote products derived from this software without\n\tspecific prior written permission.\n\n@par THIS SOFTWARE IS'` (10.1)
- `'/* included in all copies or substantial portions of the Software.       */\n/*                                                                       */\n/* THE SOFTWARE IS'` (9.7)
- `' version 2 of the License, or\n * (at your option) any later version.\n *\n * This program is'` (9.7)
- `'; either version 3 of the License, or\n   (at your option) any later version.\n\n   This program is'` (9.7)
- `' version 2 of the\n  License, or (at your option) any later version.\n  \n  This program is'` (9.6)

## latent 12063 (rate 1.78e-04)  label: ____
- `'.0 (the "License");\n//you may not use this file except in compliance with the License.'` (11.4)
- `' License, Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.'` (11.2)
- `' License, Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.'` (11.2)
- `' Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.'` (10.9)
- `', either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.'` (10.9)
- `', either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.'` (10.7)
- `', Version 2.0 (the "License");\n * you may not use this file except in compliance with the License.'` (10.7)
- `', Version 2.0 (the "License");\n * you may not use this file except in compliance with the License.'` (10.6)

## latent 25092 (rate 1.85e-04)  label: ____
- `' the License at\n *\n * http://www.apache.org/licenses/LICENSE-2.0\n *\n'` (13.2)
- `'.<|endoftext|>/*\n * The MIT License\n * Copyright © 2014-2019 Ilkka Seppälä\n *\n'` (12.8)
- `' the terms of the GNU General Public License version 2 only, as\n * published by the Free Software Foundation.\n *\n'` (12.6)
- `' the full copyright and license information, please view the LICENSE\n * file that was distributed with this source code.\n *\n'` (12.6)
- `' Software Foundation, either version 3 of the License,\n * or (at your option) any later version.\n *\n'` (12.5)
- `' * published by the Free Software Foundation, either version 3 of the License,\n * or any later version.\n *\n'` (12.3)
- `' * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU\n * Lesser General Public License for more details.\n *\n'` (12.2)
- `' Foundation,\n * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.\n *\n'` (12.1)

## latent 30733 (rate 4.02e-04)  label: ____
- `' WAY OUT OF THE USE OF THIS SOFTWARE,\n * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n *'` (8.5)
- `' Software Foundation; either\n * version 2.1 of the License, or any later version.\n *'` (8.1)
- `' express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n *'` (7.8)
- `' Free Software Foundation; either version 2 of the License, or\n * (at your option) any later version.\n *'` (7.7)
- `' on an "AS IS" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n *'` (7.7)
- `' on an "AS IS" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n *'` (7.7)
- `' ASF licenses this file\n *'` (7.6)
- `' on an "AS IS" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n *'` (7.6)

## latent 14362 (rate 7.40e-04)  label: ____
- `' driver is unloaded.\n *\n * This funtion unregisters the driver from the I2C subsystem.\n'` (14.1)
- `' *        startSymbol  |      |\n *                     |  endSymbol\n *                     |\n *                 spanStyle\n'` (14.0)
- `'\n *             |       |      |\n *        startSymbol  |      |\n *                     |  endSymbol\n'` (13.7)
- `'\n * Represents a single syntax highlighting rule.\n *\n * example:   [[ lorem ipsum ]]\n'` (13.6)
- `'aptics_rmi4_exit()\n *\n * Called by the kernel when the driver is unloaded.\n'` (13.4)
- `' the same name already exists, it is\n *   tried to become deleted. If delete fails, create fails.\n'` (13.4)
- `' string.\n *\n * This will be a string on the form "x.y.z[_string]".\n'` (13.3)
- `' * If both are equal, id is the master thread of the thread group, and is regarded as "process ID".\n'` (13.0)

## latent 22196 (rate 3.00e-04)  label: ____
- `' jobid and stores it in buf\n *\n'` (6.0)
- `' error, or if no pids are found.\n *\n'` (5.4)
- `' the driver is loaded (if a module).\n *\n'` (5.4)
- `'_tags_popular\n *\n'` (5.1)
- `'.\n *\n'` (5.1)
- `'i4_sensor_report()\n *\n'` (5.0)
- `' remain entirely with us.\n *\n'` (5.0)
- `' hwloc_bitmap_t\n *\n'` (4.9)

## latent 2532 (rate 2.56e-04)  label: ____
- `'aptics_rmi4_data\n\t\t\t\t\t'` (2.8)
- `'h(\\BSigma):=\\lim_{n\\ra\\infty}\\frac{1}{n}\\log_2 {\\rm\n'` (2.8)
- `', __VA_ARGS__)\n'` (2.8)
- `'\tdev_info(&rmi4_data->i2c_client->dev,\n\t\t\t'` (2.8)
- `'\tbreak;\n\n\t\t'` (2.8)
- `'                    <li class="languages-drawer-content__item">\n'` (2.8)
- `' retval;\n\t\t\t'` (2.8)
- `'_fhandler->fn_number == fhandler->fn_number)\n\t\t\t'` (2.8)

## latent 3928 (rate 3.67e-04)  label: ____
- `'HashTable<K, V> &h;\n        K key;\n    public:\n        proxy(HashTable'` (9.9)
- `' T cbrt_epsilon()\n{\n   typedef mpl::int_< (::std::numeric_limits'` (8.7)
- `';\n}\n\ntemplate <class T>\ninline T log_min_value(const mpl::int_'` (8.6)
- `' implements Pool<T> {\n\n    private final Pool<T> pool;\n\n    public SynchronizedPool(Pool'` (8.5)
- `' for example:\n// -max_value<double> = -1.79769e+308, max_value'` (8.4)
- `';\n}\n\ntemplate <class T>\ninline T log_min_value(const mpl::int_'` (8.4)
- `';\n}\n\ntemplate <class T>\ninline T log_min_value(const mpl::int_'` (8.1)
- `'    ordinalNumber: (...args: Array<any>) => any,\n    era: (...args: Array'` (8.0)

## latent 32274 (rate 2.42e-04)  label: ____
- `';\n            return;\n        }\n        AccountManagerHelper.get().getGoogleAccounts(new Callback<'` (10.4)
- `'.SEEDING_VALIDATING;\n        AccountManagerHelper.get().getGoogleAccounts(new Callback<'` (10.3)
- `'\n  }\n\n  void _incrementCounter() {\n    Future<'` (9.5)
- `"                 RaisedButton(\n                  child: Text('update'),\n                  onPressed: () {\n                    Future<"` (9.1)
- `'((json) => PredictionsUpdate.fromJson(json));\n\n  return update;\n\n}\n\nFuture<'` (8.9)
- `'::Duration};\n\nasync fn delay_for(pool: &ThreadPool, seconds: u64) -> Result<'` (8.8)
- `'2.9, features = ["full"]\n\nasync fn delay_for(seconds: u64) -> Result<'` (8.8)
- `' _MyHomePageState extends State<MyHomePage> {\n  int _counter = 0;\n\n  Future<'` (8.6)

## latent 19617 (rate 6.50e-05)  label: ____
- `'_EXPLICIT_TEMPLATE_TYPE(T))\n{\n   return std::numeric_limits<'` (8.2)
- `'int_< (::std::numeric_limits<T>::radix == 2) ? std::numeric_limits<'` (8.1)
- `'_COMPILE_TIME_CONSTANTS\n   BOOST_STATIC_ASSERT( ::std::numeric_limits<'` (8.0)
- `'_COMPILE_TIME_CONSTANTS\n   BOOST_STATIC_ASSERT( ::std::numeric_limits<'` (8.0)
- `'_CONSTANTS\n   return detail::epsilon<T>(mpl::bool_< ::std::numeric_limits<'` (8.0)
- `' forth_root_epsilon()\n{\n   typedef mpl::int_< (::std::numeric_limits<'` (7.8)
- `'_MATH_STD_USING\n   static const T r_eps = sqrt(sqrt(tools::epsilon<'` (7.8)
- `'numeric_limits<T>::is_specialized>());\n#else\n   return ::std::numeric_limits<'` (7.8)

## latent 17986 (rate 4.79e-05)  label: ____
- `'6773604745549542403508090496e-12L);\n}\n\ntemplate <class'` (12.5)
- `'1917516667460937040007436176452688944747L);\n}\n\ntemplate <class'` (12.4)
- `'8989272793696693569753008995e-6L);\n}\n\ntemplate <class'` (12.4)
- `'    ~HashTable();\n};\n\nAnd this is the implementation of each overload of the brackets:\ntemplate <typename'` (12.3)
- `'0567321328132261891067079047605e-4L);\n}\n\ntemplate <class'` (12.3)
- `'33569506281281311031656150598474e-9L);\n}\n\ntemplate <class'` (12.2)
- `'TEMPLATE_TYPE(T))\n{\n   return 709.0;\n}\n\ntemplate <class'` (12.0)
- `'eps = sqrt(tools::epsilon<T>());\n   return r_eps;\n}\n\ntemplate <class'` (12.0)
