using Microsoft.AspNetCore.Mvc;
using System;
using System;
using System.Collections.Generic;
using ILGPU;
using ILGPU.Runtime;
using GDvsCD.Services;
using System.IO;
using System.Text.Json;
namespace GDvsCD.Controllers
{
  
    [ApiController]
    [Route("[controller]")]
    public class CalcController : ControllerBase
    {
        private readonly ILogger<CalcController> _logger;

        public CalcController(ILogger<CalcController> logger)
        {
            _logger = logger;
        }



     [HttpGet("[action]")]
        public async Task<IActionResult> Get(float eta,int numIter)
        {
            // Create an ILGPU context.
            using var context = Context.CreateDefault();
            // Get the first available accelerator (e.g. a Cuda or OpenCL GPU).
            using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

            // Read the JSON file content.
            string json = await System.IO.File.ReadAllTextAsync("./Dataset/features.json");

            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            double[,] X = Algo.ConvertJaggedTo2D(JsonSerializer.Deserialize<double[][]>(json, options));




             json = await System.IO.File.ReadAllTextAsync("./Dataset/output.json");


            double[] y = JsonSerializer.Deserialize<double[]>(json, options);
            // y is the target vector.

            // Initial guess for theta (n parameters); here two parameters.
            double[] theta = new double[] {1, 1,1,1,1,1,1,1 };

            // Run coordinate descent.
            var resultCD = Algo.CoordinateDescent(theta, X, y, accelerator, alpha: eta, numIters: numIter);
            var resultGD = Algo.GradientDescent(theta, X, y, accelerator, alpha: eta, numIters: numIter);

            // Dispose of the accelerator (automatically done via using-statement).
            accelerator.Dispose();



            return Ok(new {
            CD = new { costHistory=resultCD.costHistory,theta= resultCD .theta},
                GD = new { costHistory = resultGD.costHistory, theta = resultGD.theta }
            });
        }
    }


}
