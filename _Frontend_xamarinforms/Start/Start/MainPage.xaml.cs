using Plugin.Media;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xamarin.Forms;
using Xamarin.Essentials;
using System.IO;
using System.Net.Http;
using System.Threading;
using Newtonsoft.Json;
using System.Net;
using NAudio.Wave;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision.Models;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision;

namespace Start
{
    public class Prediction
    {
        public string TagId { get; set; }
        public string Tag { get; set; }
        public double Probability { get; set; }
    }
    public class Response
    {
        public string Id { get; set; }
        public string Project { get; set; }
        public string Iteration { get; set; }
        public DateTime Created { get; set; }
        public List<Prediction> Predictions { get; set; }
    }
    public class Language
    {
        public string selectedlanguage;
    }
    public partial class MainPage : ContentPage
    {
        private string subscriptionKey = "Connection String";
        private string ausgewerterSatz;
        private bool CheckWarning = true;
        public MainPage()
        {
            InitializeComponent();
            //Vibration.Vibrate();
            Device.StartTimer(TimeSpan.FromSeconds(15), () =>
            {
                MakePhotoAsync();      
                return true;
            });
        }
        public async Task MakePhotoAsync()
        {
            await CrossMedia.Current.Initialize();
            if (!CrossMedia.Current.IsCameraAvailable || !CrossMedia.Current.IsTakePhotoSupported)
            {
                await TextToSpeech.SpeakAsync("No Camera found");
                return;
            }
            var file = await CrossMedia.Current.TakePhotoAsync(new Plugin.Media.Abstractions.StoreCameraMediaOptions
            {
                Directory = "Sample",
                Name = "test.jpg"
            });
            if (file == null)
                return;
            List<VisualFeatureTypes> features =
            new List<VisualFeatureTypes>()
            {
            VisualFeatureTypes.Categories, VisualFeatureTypes.Description,
            VisualFeatureTypes.Faces,
            VisualFeatureTypes.Tags
            };
            ComputerVisionClient computerVision = new ComputerVisionClient(
                new ApiKeyServiceClientCredentials(subscriptionKey),
                new System.Net.Http.DelegatingHandler[] { });
            computerVision.Endpoint = "https://francecentral.api.cognitive.microsoft.com/";
            using (Stream imageStream = File.OpenRead(file.Path))
            {
                ImageAnalysis analysis = await computerVision.AnalyzeImageInStreamAsync(
                imageStream, features);
                ausgewerterSatz = analysis.Description.Captions[0].Text;
            }
            await TextToSpeech.SpeakAsync(ausgewerterSatz);
            ausgewerterSatz = "";
            Stream stream = file.GetStream();
            var imageBytes = GetImageAsByteData(stream);
            FileInfo fil = new FileInfo(file.Path);
            //Upload(imageBytes, fil.Name); Would Post to Vits Rest API
        }
        private async Task MakePredictionAsync(Stream stream) //Calls Custom Vision 
        {
            var imageBytes = GetImageAsByteData(stream);
            var url = "https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/8dde27e7-f188-4528-8874-087881f4a1c6/image";
            using (HttpClient client = new HttpClient())
            {
                client.DefaultRequestHeaders.Add("Prediction-Key", "<>");

                using (var content = new ByteArrayContent(imageBytes))
                {
                    content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/octet-stream");
                    var response = await client.PostAsync(url, content);
                    var responseString = await response.Content.ReadAsStringAsync();

                    var predictions = JsonConvert.DeserializeObject<Response>(responseString);
                    var resp = predictions.Predictions[0];
                    if(resp.TagId == "2ba3cbf9-a02b-47c1-9f32-a487d6c7cdce")
                    {
                        TextToSpeech.SpeakAsync("Police is on the Way");
                    }
                }
            }
        }
        private byte[] GetImageAsByteData(Stream stream)
        {
            BinaryReader binaryReader = new BinaryReader(stream);
            return binaryReader.ReadBytes((int)stream.Length);
        }
        public async void Upload(byte[] bytes, string fileName)
        {
            ASCIIEncoding encoding = new ASCIIEncoding();
            Uri webService = new Uri("http://e60a6b3c.ngrok.io/handshake");
            HttpRequestMessage requestMessage = new HttpRequestMessage(HttpMethod.Post, webService);
            requestMessage.Headers.ExpectContinue = false;
            MultipartFormDataContent multiPartContent = new MultipartFormDataContent("----MyGreatBoundary");
            ByteArrayContent byteArrayContent = new ByteArrayContent(bytes);
            byteArrayContent.Headers.Add("Content-Type", "multipart/form-data");
            multiPartContent.Add(byteArrayContent, "file", fileName);
            requestMessage.Content = multiPartContent;
            HttpClient httpClient = new HttpClient();
            Task<HttpResponseMessage> httpRequest = httpClient.SendAsync(requestMessage, HttpCompletionOption.ResponseContentRead, CancellationToken.None);
            HttpResponseMessage httpResponse = httpRequest.Result;
            string b = await httpResponse.Content.ReadAsStringAsync();
            string location = httpResponse.Headers.Location.ToString();
            await TextToSpeech.SpeakAsync(location);
            //var x = await httpResponse.Content.ReadAsStreamAsync();
            //var hp = new WebClient();
            //var htp = hp.DownloadData("http://eb9826b0.ngrok.io/speech.mp3");
            //File.WriteAllBytes("somefile.mp3", htp);
            //var player = Plugin.SimpleAudioPlayer.CrossSimpleAudioPlayer.Current;
            //player.Load("somefile.mp3");
            //player.Play();
        }
    }
}
